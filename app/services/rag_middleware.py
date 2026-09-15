"""ARM-friendly async RAG adapter for the attached SignalMinds core."""

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from groq import AsyncGroq

from app.core.config import get_settings

logger = logging.getLogger(__name__)
CORE_DIRECTORY = Path(__file__).resolve().parents[1] / "rag_core"
TEXTBOOK_DIRECTORY = CORE_DIRECTORY / "data" / "textbooks"
DOCUMENT_CACHE_PATH = CORE_DIRECTORY / "data" / "rag_text_cache.json"
DOCUMENT_CACHE_VERSION = 1
_core_lock = Lock()
_llm: Any | None = None
_build_rag_prompt = None
_documents: list[dict[str, Any]] | None = None
_documents_origin = "memory"
_sessions: dict[str, list[dict[str, str]]] = {}
_groq_client: AsyncGroq | None = None
_groq_client_lock = asyncio.Lock()

LANGUAGE_PROFILES: dict[str, dict[str, str]] = {
    "hi-IN": {"name": "Hindi", "style": "Hinglish (conversational Hindi written in Latin/English script)", "example": "CPU scheduling ek mechanism hai jisme operating system processes ko CPU time allocate karta hai, jaise Round Robin aur FCFS algorithms.", "fallback": "Sorry, abhi answer fetch nahi ho paya. Please phir se poochiye."},
    "te-IN": {"name": "Telugu", "style": "Tinglish (conversational Telugu written in Latin/English script)", "example": "CPU scheduling ante operating system lo processes ki CPU time allocate chese mechanism, like FCFS mariyu Round Robin algorithms.", "fallback": "Sorry, ippudu answer fetch avvaledu. Please malli adagandi."},
    "ta-IN": {"name": "Tamil", "style": "Tanglish (conversational Tamil written in Latin/English script)", "example": "Deadlock na operating system-la rendu processes resource kaaga wait panni block aaguradhu, idhai Banker's algorithm use panni handle panlaam.", "fallback": "Sorry, ippo answer fetch panna mudiyala. Please marubadiyum kelunga."},
    "kn-IN": {"name": "Kannada", "style": "Kanglish (conversational Kannada written in Latin/English script)", "example": "CPU scheduling andre operating system-alli processes ge CPU time allocate maduva mechanism, like Round Robin mathu FCFS algorithms.", "fallback": "Sorry, ivaga answer fetch agalilla. Please matte keli."},
    "ml-IN": {"name": "Malayalam", "style": "Manglish (conversational Malayalam written in Latin/English script)", "example": "CPU scheduling ennal operating system-il processes-inu CPU time allocate cheyyunna mechanism aanu, like Round Robin algorithms.", "fallback": "Sorry, ippo answer fetch cheyyan pattiyilla. Please veendum chodikku."},
    "mr-IN": {"name": "Marathi", "style": "Conversational Marathi in Latin/English script mixed with English", "example": "CPU scheduling mhanje operating system madhye processes na CPU time allocate karnari mechanism, jase Round Robin algorithms.", "fallback": "Sorry, ata answer fetch zala nahi. Please punha vichara."},
    "bn-IN": {"name": "Bengali", "style": "Benglish (conversational Bengali written in Latin/English script)", "example": "Virtual memory operating system er emon ekta technique jekhane RAM kom thakleo secondary storage ke main memory hishebe use kora hoy.", "fallback": "Sorry, ekhon answer fetch kora jayni. Please abar jiggesh korun."},
    "gu-IN": {"name": "Gujarati", "style": "Conversational Gujarati in Latin/English script mixed with English", "example": "CPU scheduling etle operating system ma processes ne CPU time allocate karvano mechanism, jem ke Round Robin algorithms.", "fallback": "Sorry, atyare answer fetch thai shakyo nathi. Please fari pucho."},
    "pa-IN": {"name": "Punjabi", "style": "Conversational Punjabi in Latin/English script mixed with English", "example": "CPU scheduling ik mechanism hai jis naal operating system processes nu CPU time allocate karda hai, jiwe Round Robin algorithms.", "fallback": "Sorry, hun answer fetch nahi ho sakya. Please phir pucho."},
    "od-IN": {"name": "Odia", "style": "Conversational Odia in Latin/English script mixed with English", "example": "CPU scheduling heuchi operating system re processes ku CPU time allocate kariba mechanism, jemiti Round Robin algorithms.", "fallback": "Sorry, ebe answer fetch heiparila nahi. Please puni pacharantu."},
    "en-IN": {"name": "English", "style": "Clear, concise Indian English", "example": "CPU scheduling is the operating system mechanism that allocates CPU time to ready processes using algorithms like FCFS and Round Robin.", "fallback": "Sorry, I could not fetch the answer. Please ask again."},
}
DEFAULT_PROFILE: dict[str, str] = {
    "name": "Telugu",
    "style": "Tinglish (conversational Telugu written in Latin/English script)",
    "example": "CPU scheduling ante operating system lo processes ki CPU time allocate chese mechanism, like FCFS mariyu Round Robin algorithms.",
    "fallback": "Sorry, ippudu answer fetch avvaledu. Please malli adagandi.",
}
_LANGUAGE_PROFILES_BY_NORMALIZED_CODE = {code.lower(): profile for code, profile in LANGUAGE_PROFILES.items()}

VERNACULAR_KEYWORD_PATTERNS = {
    "te-IN": re.compile(r"\b(ante|enti|ela|mariyu|lo|unna|chese|cheppandi|cheyadam|kadha|gurinchi|enduku|chesukondi)\b", re.IGNORECASE),
    "ta-IN": re.compile(r"\b(enna|epdi|adha|pannuvanga|solunga|la|kaaga|irukku|enna-na|panlaam)\b", re.IGNORECASE),
    "bn-IN": re.compile(r"\b(ki|bhabe|kaaj|kore|bolun|ekta|jekhane|kora|hoy)\b", re.IGNORECASE),
    "hi-IN": re.compile(r"\b(kya|kaise|hota|hoti|hai|bataiye|batao|karte|jisme|karega)\b", re.IGNORECASE),
}


VOICE_AGENT_SYSTEM_PROMPT = """
You are an ultra-fast, conversational AI voice tutor answering live phone calls for rural students ("Shiksha Vani").

STRICT CRITICAL RULES:
1. Script & Transliteration Requirement:
   - OUTPUT ONLY IN PLAIN ASCII LATIN/ENGLISH SCRIPT (Roman transliteration).
   - NEVER output native Indic scripts (NO Telugu script తెలుగు, NO Devanagari हिन्दी, NO Tamil script தமிழ், NO Bengali script বাংলা).
   - The phone TTS system expects pure Roman alphabet Latin text (e.g., "ante", "cheppandi", "karega", "panlaam").

2. Natural Code-Mixed Vernacular + English Technical Terms:
   - When speaking regional languages (Telugu, Tamil, Hindi, Bengali, etc.), speak in NATURAL CONVERSATIONAL CODE-MIXED style (e.g., Tinglish, Tanglish, Hinglish, Benglish).
   - ALL academic, technical, IT, scientific, and computing terms MUST REMAIN IN ENGLISH. Retain words like: CPU, Scheduling, Process, Thread, Operating System, Memory, RAM, Virtual Memory, Deadlock, Algorithm, Round Robin, FCFS, Photosynthesis, Database, Cache, Hardware.
   - NEVER translate technical concepts into complex or archaic native words.
   - Examples of desired conversational code-mixed vernacular:
     * Telugu (Tinglish): "CPU scheduling ante operating system lo processes ki CPU time allocate chese mechanism, like FCFS mariyu Round Robin algorithms."
     * Tamil (Tanglish): "Deadlock na operating system-la rendu processes resource kaaga wait panni block aaguradhu, idhai Banker's algorithm use panni handle panlaam."
     * Hindi (Hinglish): "CPU scheduling ek mechanism hai jisme operating system processes ko CPU time allocate karta hai, jaise Round Robin aur FCFS algorithms."
     * Bengali (Benglish): "Virtual memory operating system er emon ekta technique jekhane RAM kom thakleo secondary storage ke main memory hishebe use kora hoy."

3. Spoken Brevity and Phone Call Format:
   - Keep answers strictly to 1 to 2 concise spoken sentences (25 to 35 words maximum).
   - Make it sound warm, encouraging, and natural for a telephone conversation.
   - Never stop mid-thought; always finish the final sentence cleanly with punctuation (. or !).
   - ABSOLUTELY NO bullet points, lists, numbered items, Markdown formatting, asterisks, or emojis.
""".strip()


def get_language_profile(language_code: str | None) -> dict[str, str]:
    """Return a case-insensitive caller-language profile, with a safe default."""
    normalized = str(language_code or "").strip().lower()
    return _LANGUAGE_PROFILES_BY_NORMALIZED_CODE.get(normalized, DEFAULT_PROFILE)


def _language_style_instruction(language_code: str) -> str:
    """Tell the LLM exactly how the phone caller should hear the answer."""
    profile = get_language_profile(language_code)
    example = profile.get("example", "")
    if profile["name"] == "English":
        settings = get_settings()
        default_lang = getattr(settings, "default_caller_language", "te-IN")
        if default_lang and default_lang != "en-IN":
            def_profile = get_language_profile(default_lang)
            return (
                f"The caller is a regional student who may ask questions using English technical words. "
                f"Respond in natural conversational {def_profile['style']} where all technical terms remain in English. "
                f"MANDATORY: Output ONLY in Roman/Latin script (ASCII English letters), NEVER in native script! "
                f"Example response style: \"{def_profile.get('example', '')}\". Keep to 1-2 spoken sentences."
            )
        return "The caller spoke English. Respond in concise, clear English in 1-2 spoken sentences."
    return (
        f"DETECTED CALLER LANGUAGE OVERRIDES the translated English query: {profile['name']} ({language_code}). "
        f"You MUST respond in {profile['style']}. "
        f"MANDATORY: Output ONLY in Roman/Latin script (ASCII English letters), NEVER in native script! "
        f"Keep ALL academic, computing, and technical terms in English (e.g., CPU, Process, Scheduling, Deadlock, Algorithm, Memory). "
        f"Use natural conversational words for everything else (e.g. for Telugu use 'ante', 'lo', 'chese', 'mariyu'). "
        f"Example response style: \"{example}\". "
        f"Keep response to 1-2 concise spoken sentences (under 35 words)."
    )


async def _get_groq_client() -> AsyncGroq:
    """Return one process-wide client so fallback calls reuse TCP/TLS sessions."""
    global _groq_client
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    async with _groq_client_lock:
        if _groq_client is None:
            _groq_client = AsyncGroq(api_key=settings.groq_api_key)
        return _groq_client


async def close_groq_client() -> None:
    """Close the shared Groq HTTP session during FastAPI shutdown."""
    global _groq_client
    async with _groq_client_lock:
        client, _groq_client = _groq_client, None
    if client is not None:
        await client.close()


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    return [
        text[index : index + chunk_size].strip()
        for index in range(0, len(text), chunk_size - overlap)
        if text[index : index + chunk_size].strip()
    ]


def _document_signature() -> list[dict[str, int | str]]:
    """Identify the PDFs used to build the persisted lightweight text cache."""
    return [
        {
            "name": pdf_path.name,
            "size": pdf_path.stat().st_size,
            "mtime_ns": pdf_path.stat().st_mtime_ns,
        }
        for pdf_path in sorted(TEXTBOOK_DIRECTORY.glob("*.pdf"))
    ]


def _read_document_cache(signature: list[dict[str, int | str]]) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(DOCUMENT_CACHE_PATH.read_text(encoding="utf-8"))
        documents = payload.get("documents") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != DOCUMENT_CACHE_VERSION
            or payload.get("signature") != signature
            or not isinstance(documents, list)
            or not documents
        ):
            return None
        if not all(isinstance(item, dict) and isinstance(item.get("text"), str) for item in documents):
            return None
        return documents
    except (OSError, ValueError, TypeError):
        return None


def _write_document_cache(signature: list[dict[str, int | str]], documents: list[dict[str, Any]]) -> None:
    """Atomically persist extracted text so future server starts skip PDF parsing."""
    try:
        DOCUMENT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = DOCUMENT_CACHE_PATH.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "version": DOCUMENT_CACHE_VERSION,
                    "signature": signature,
                    "documents": documents,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(DOCUMENT_CACHE_PATH)
    except OSError:
        logger.warning("rag_text_cache_write_failed", extra={"event": "rag"})


def _load_documents() -> list[dict[str, Any]]:
    global _documents, _documents_origin
    if _documents is not None:
        return _documents
    signature = _document_signature()
    cached_documents = _read_document_cache(signature)
    if cached_documents is not None:
        _documents = cached_documents
        _documents_origin = "disk_cache"
        return _documents
    from pypdf import PdfReader

    documents: list[dict[str, Any]] = []
    for pdf_path in TEXTBOOK_DIRECTORY.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            for text in _chunk_text(page.extract_text() or ""):
                documents.append(
                    {"text": text, "book": pdf_path.stem, "page": page_number}
                )
    if not documents:
        raise RuntimeError("No readable textbooks are available for RAG retrieval")
    _documents = documents
    _documents_origin = "pdf_parse"
    _write_document_cache(signature, documents)
    return documents


def _retrieve(question: str, limit: int = 5) -> tuple[list[dict[str, Any]], int]:
    query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    if not query_terms:
        return [], 0
    scored: list[tuple[int, dict[str, Any]]] = []
    for document in _load_documents():
        words = re.findall(r"[a-z0-9]+", document["text"].lower())
        score = sum(words.count(term) for term in query_terms)
        if score:
            scored.append((score, document))
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    return [document for _, document in ranked[:limit]], (ranked[0][0] if ranked else 0)


def _load_llm() -> tuple[Any, Any]:
    global _llm, _build_rag_prompt
    if _llm is not None and _build_rag_prompt is not None:
        return _llm, _build_rag_prompt
    if not CORE_DIRECTORY.is_dir():
        raise RuntimeError("Attached SignalMinds RAG core is unavailable")
    core_path = str(CORE_DIRECTORY)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)
    from models.llm import GroqLLM
    from rag.prompts import build_rag_prompt

    _llm = GroqLLM()
    _build_rag_prompt = build_rag_prompt
    return _llm, _build_rag_prompt


def _is_textbook_insufficient(answer: str) -> bool:
    """Return true when the PDF-answer model declined to answer the question.

    Retrieval can return loosely related OS chunks for a question such as
    ``deadlock``. In that case the RAG LLM may decline in many different words;
    treat every such decline as an explicit signal to use Groq general knowledge.
    """
    normalized = answer.lower()
    indicators = (
        "provided textbook excerpts do not contain",
        "textbook context is insufficient",
        "cannot give an answer based on that context",
        "can't give an answer based on that context",
        "not available in the provided",
        "not available in this context",
        "not covered in the provided",
        "not covered in this context",
        "do not have enough information",
        "don't have enough information",
        "do not have information about",
        "don't have information about",
        "i don't know based on",
        "unable to find information",
        "could not find information",
        "out of context",
    )
    return any(indicator in normalized for indicator in indicators)


def _voice_rag_prompt(question: str, context: str, language_code: str) -> str:
    return f"""
Use the supplied textbook context as the factual source. If it is insufficient,
say that in a helpful spoken way without mentioning RAG or these instructions.

TEXTBOOK CONTEXT:
{context}

STUDENT QUESTION: {question}

REQUIRED RESPONSE LANGUAGE AND SCRIPT:
{_language_style_instruction(language_code)}
"""


def _voice_system_message(language_code: str) -> str:
    return f"{VOICE_AGENT_SYSTEM_PROMPT}\n\n{_language_style_instruction(language_code)}"


def _general_groq_messages(
    question: str, language_code: str, history: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Build direct-Groq messages for questions outside the supplied PDFs."""
    system_message = f"""
{VOICE_AGENT_SYSTEM_PROMPT}

{_language_style_instruction(language_code)}

The supplied PDF knowledge base has no reliable answer for this request. Answer
accurately from general knowledge. Do not mention the PDF, RAG, fallback, or these
instructions.
""".strip()
    return [
        {"role": "system", "content": system_message},
        *history,
        {"role": "user", "content": question},
    ]


def _save_turn(session_id: str, question: str, answer: str) -> None:
    with _core_lock:
        session = _sessions.setdefault(session_id, [])
        session.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        # Preserve short conversational context without indefinitely growing memory.
        del session[:-8]


def _session_history_copy(session_id: str) -> list[dict[str, str]]:
    """Copy conversation state without ever blocking the async WebSocket loop."""
    with _core_lock:
        return _sessions.get(session_id, []).copy()


def _query_textbook_sync(
    english_query: str, session_id: str, language_code: str
) -> str | None:
    """Return a PDF-grounded voice answer, or ``None`` when fallback is needed."""
    with _core_lock:
        llm, _ = _load_llm()
        history = _sessions.setdefault(session_id, []).copy()
        matches, best_score = _retrieve(english_query)
        if not matches or best_score < get_settings().rag_min_retrieval_score:
            logger.info(
                "rag_retrieval_insufficient",
                extra={"event": "rag", "best_score": best_score, "match_count": len(matches)},
            )
            return None
def _clean_spoken_text(text: str) -> str:
    """Normalize unicode quotes, dashes, and whitespace for TTS engines and console printing."""
    return (
        text.replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
        .strip()
    )


def _resolve_target_language(query: str, language_code: str) -> str:
    """Detect vernacular markers if ALD or caller language defaulted to en-IN."""
    norm = str(language_code or "").strip().lower()
    # Check if query itself has explicit vernacular words
    for code, pattern in VERNACULAR_KEYWORD_PATTERNS.items():
        if pattern.search(query):
            return code
    if norm in {"", "en-in", "en"}:
        settings = get_settings()
        default_lang = getattr(settings, "default_caller_language", "te-IN")
        if default_lang and default_lang.lower() not in {"en-in", "en"}:
            return default_lang
    return language_code or "te-IN"


def _query_textbook_sync(
    english_query: str, session_id: str, language_code: str
) -> str | None:
    """Return a PDF-grounded voice answer, or ``None`` when fallback is needed."""
    target_lang = _resolve_target_language(english_query, language_code)
    with _core_lock:
        llm, _ = _load_llm()
        history = _sessions.setdefault(session_id, []).copy()
        matches, best_score = _retrieve(english_query)
        if not matches or best_score < get_settings().rag_min_retrieval_score:
            logger.info(
                "rag_retrieval_insufficient",
                extra={"event": "rag", "best_score": best_score, "match_count": len(matches)},
            )
            return None
        context = "\n".join(
            f"Book: {match['book']}\nPage: {match['page']}\nContent:\n{match['text']}"
            for match in matches
        )
        # Do not use the generic core prompt: this response must already be in
        # the caller's spoken style so no mechanical translation is necessary.
        answer = llm.generate(
            _voice_rag_prompt(english_query, context, target_lang),
            history,
            system_prompt=_voice_system_message(target_lang),
        )
        if not isinstance(answer, str) or not answer.strip() or _is_textbook_insufficient(answer):
            logger.info("rag_answer_insufficient", extra={"event": "rag"})
            return None
    answer = _clean_spoken_text(answer)
    _save_turn(session_id, english_query, answer)
    return answer


async def _query_groq_fallback(
    english_query: str, session_id: str, language_code: str
) -> str | None:
    """Call Groq directly only after retrieval has no usable PDF context."""
    target_lang = _resolve_target_language(english_query, language_code)
    settings = get_settings()
    if not settings.groq_api_key:
        logger.error("groq_fallback_unconfigured", extra={"event": "groq_fallback"})
        return None
    history = await asyncio.to_thread(_session_history_copy, session_id)
    loop = asyncio.get_running_loop()
    started = loop.time()
    logger.info("[GROQ FALLBACK START] Querying general knowledge.", extra={"event": "groq_fallback"})
    try:
        client = await _get_groq_client()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.groq_model,
                messages=_general_groq_messages(english_query, target_lang, history),
                temperature=0.2,
                # GPT-OSS otherwise spends the small voice budget on hidden
                # reasoning and can return an empty final message.
                reasoning_effort="low",
                include_reasoning=False,
                max_tokens=100,
            ),
            timeout=settings.rag_query_timeout_seconds,
        )
        answer = response.choices[0].message.content
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Groq fallback returned no answer")
        answer = _clean_spoken_text(answer)
        await asyncio.to_thread(_save_turn, session_id, english_query, answer)
        duration = loop.time() - started
        logger.info(
            "[GROQ FALLBACK FINISH] Completed in %.2fs.",
            duration,
            extra={"event": "groq_fallback", "duration_seconds": round(duration, 3)},
        )
        return answer
    except Exception:
        logger.exception("groq_fallback_failed", extra={"event": "groq_fallback"})
        return None


async def query_rag(
    english_query: str, session_id: str, language_code: str = "en-IN"
) -> str | None:
    """Return a caller-style PDF answer, falling back to direct Groq if needed."""
    resolved_lang = _resolve_target_language(english_query, language_code)
    try:
        textbook_answer = await asyncio.wait_for(
            asyncio.to_thread(_query_textbook_sync, english_query, session_id, resolved_lang),
            timeout=get_settings().rag_query_timeout_seconds,
        )
        if textbook_answer:
            return textbook_answer
    except TimeoutError:
        logger.warning("rag_query_timed_out", extra={"event": "rag"})
    except Exception:
        logger.exception("rag_query_failed", extra={"event": "rag"})
    return await _query_groq_fallback(english_query, session_id, resolved_lang)


async def warm_rag_index() -> None:
    """Pre-parse the textbook in the background so the first call is faster."""
    try:
        await asyncio.to_thread(_warm_rag_index_sync)
        logger.info(
            "rag_index_warmed",
            extra={"event": "rag", "status": _documents_origin},
        )
    except Exception:
        logger.exception("rag_index_warm_failed", extra={"event": "rag"})


def _warm_rag_index_sync() -> None:
    with _core_lock:
        _load_documents()
