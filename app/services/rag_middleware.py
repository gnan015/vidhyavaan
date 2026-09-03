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

VOICE_AGENT_SYSTEM_PROMPT = """
You are an intelligent, friendly AI voice assistant answering phone calls.
Follow these strict conversational voice guidelines:
1. Brevity: Answer in 1 to 2 short, direct sentences (maximum 35-40 words). The response will be spoken aloud over a phone call.
2. Language style:
   - If the user speaks Telugu, respond in natural spoken Tinglish (conversational Telugu blended with English).
   - NEVER translate technical concepts, computer science terms, or modern everyday nouns into pure or formal Telugu (e.g., keep "Operating System", "CPU", "Internet", "Memory" as English words).
   - The pronunciation and flow should sound like a modern urban conversation.
   - If the user speaks English, respond in clear, concise English.
3. No formatting: Do not include asterisks, bullet points, Markdown, emojis, or code blocks, as text will be read directly by Text-to-Speech.
""".strip()


def _language_style_instruction(language_code: str) -> str:
    """Tell the LLM exactly how the phone caller should hear the answer."""
    normalized = language_code.lower()
    if normalized.startswith("te"):
        return (
            "The caller spoke Telugu. Respond in natural spoken Tinglish, using "
            "Roman/English letters as in everyday conversation. Keep all technical "
            "terms, IT words, programming concepts, acronyms, and modern nouns in "
            "English. Do not use formal or textbook Telugu translations."
        )
    if normalized.startswith("en"):
        return "The caller spoke English. Respond in concise, clear English."
    return (
        f"The caller language is {language_code}. Respond conversationally in that "
        "language while keeping technical terms, IT words, programming concepts, "
        "acronyms, and modern nouns in English."
    )


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
    normalized = answer.lower()
    indicators = (
        "provided textbook excerpts do not contain",
        "textbook context is insufficient",
        "cannot give an answer based on that context",
        "can't give an answer based on that context",
    )
    return any(indicator in normalized for indicator in indicators)


def _voice_rag_prompt(question: str, context: str, language_code: str) -> str:
    return f"""
Use the supplied textbook context as the factual source. If it is insufficient,
say that in a helpful spoken way without mentioning RAG or these instructions.

TEXTBOOK CONTEXT:
{context}

STUDENT QUESTION: {question}
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
        context = "\n".join(
            f"Book: {match['book']}\nPage: {match['page']}\nContent:\n{match['text']}"
            for match in matches
        )
        # Do not use the generic core prompt: this response must already be in
        # the caller's spoken style so no mechanical translation is necessary.
        answer = llm.generate(
            _voice_rag_prompt(english_query, context, language_code),
            history,
            system_prompt=_voice_system_message(language_code),
        )
        if not isinstance(answer, str) or not answer.strip() or _is_textbook_insufficient(answer):
            logger.info("rag_answer_insufficient", extra={"event": "rag"})
            return None
    answer = answer.strip()
    _save_turn(session_id, english_query, answer)
    return answer


async def _query_groq_fallback(
    english_query: str, session_id: str, language_code: str
) -> str | None:
    """Call Groq directly only after retrieval has no usable PDF context."""
    settings = get_settings()
    if not settings.groq_api_key:
        logger.error("groq_fallback_unconfigured", extra={"event": "groq_fallback"})
        return None
    history = await asyncio.to_thread(_session_history_copy, session_id)
    loop = asyncio.get_running_loop()
    started = loop.time()
    logger.info("[GROQ FALLBACK START] Querying general knowledge.", extra={"event": "groq_fallback"})
    try:
        client = AsyncGroq(api_key=settings.groq_api_key)
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.groq_model,
                messages=_general_groq_messages(english_query, language_code, history),
                temperature=0.2,
                max_tokens=140,
            ),
            timeout=settings.rag_query_timeout_seconds,
        )
        answer = response.choices[0].message.content
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Groq fallback returned no answer")
        answer = answer.strip()
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
    try:
        textbook_answer = await asyncio.wait_for(
            asyncio.to_thread(_query_textbook_sync, english_query, session_id, language_code),
            timeout=get_settings().rag_query_timeout_seconds,
        )
        if textbook_answer:
            return textbook_answer
    except TimeoutError:
        logger.warning("rag_query_timed_out", extra={"event": "rag"})
    except Exception:
        logger.exception("rag_query_failed", extra={"event": "rag"})
    return await _query_groq_fallback(english_query, session_id, language_code)


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
