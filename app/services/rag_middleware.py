"""
PM-AJAY Virtual Livelihood Assistant — RAG & LLM middleware.

Replaces legacy Shiksha Vani academic-tutor middleware.
Answers caller questions about PM-AJAY livelihood schemes, skill training,
asset subsidies, credit linkages, and nearby training centers using
ChromaDB vector search + Groq LLM generation.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any

from groq import AsyncGroq

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_core_lock = Lock()
_sessions: dict[str, list[dict[str, str]]] = {}
_groq_client: AsyncGroq | None = None
_groq_client_lock = asyncio.Lock()


# ─── Language Profiles ────────────────────────────────────────────────────────

LANGUAGE_PROFILES: dict[str, dict[str, str]] = {
    "te-IN": {
        "name": "Telugu",
        "script_name": "Telugu script (తెలుగు)",
        "teacher_dialect": "neutral, polite, standard conversational Telugu",
        "native_example": "PM-AJAY scheme లో SC families కి skill training, asset subsidy, మరియు NSFDC loan అందుబాటులో ఉంటాయి.",
        "native_fallback": "క్షమించండి, ప్రస్తుతం సమాధానం పొందలేకపోయాను. దయచేసి మళ్ళీ అడగండి.",
        "roman_fallback": "Sorry, ippudu answer fetch avvaledu. Please malli adagandi.",
    },
    "ta-IN": {
        "name": "Tamil",
        "script_name": "Tamil script (தமிழ்)",
        "teacher_dialect": "neutral, polite, standard conversational Tamil",
        "native_example": "PM-AJAY scheme-ல் SC பயனாளிகளுக்கு skill training, asset subsidy மற்றும் NSFDC loan கிடைக்கும்.",
        "native_fallback": "மன்னிக்கவும், இப்போது பதில் பெற முடியவில்லை. தயவுசெய்து மீண்டும் கேளுங்கள்.",
        "roman_fallback": "Sorry, ippo answer fetch panna mudiyala. Please marubadiyum kelunga.",
    },
    "hi-IN": {
        "name": "Hindi",
        "script_name": "Devanagari script (हिन्दी)",
        "teacher_dialect": "clear, polite, standard conversational Hindi",
        "native_example": "PM-AJAY scheme में SC परिवारों को free skill training, asset subsidy और NSFDC loan मिलता है।",
        "native_fallback": "माफ़ कीजिए, अभी उत्तर नहीं मिल पाया। कृपया फिर से पूछिए।",
        "roman_fallback": "Sorry, abhi answer fetch nahi ho paya. Please phir se poochiye.",
    },
    "bn-IN": {
        "name": "Bengali",
        "script_name": "Bengali script (বাংলা)",
        "teacher_dialect": "clean, polite, standard conversational Bengali",
        "native_example": "PM-AJAY scheme এ SC পরিবারগুলো free skill training, asset subsidy এবং NSFDC loan পাবেন।",
        "native_fallback": "দুঃখিত, এখন উত্তর পাওয়া যায়নি। দয়া করে আবার জিজ্ঞাসা করুন।",
        "roman_fallback": "Sorry, ekhon answer fetch kora jayni. Please abar jiggesh korun.",
    },
    "kn-IN": {
        "name": "Kannada",
        "script_name": "Kannada script (ಕನ್ನಡ)",
        "teacher_dialect": "clear, polite, standard conversational Kannada",
        "native_example": "PM-AJAY scheme ನಲ್ಲಿ SC ಕುಟುಂಬಗಳಿಗೆ free skill training, asset subsidy ಮತ್ತು NSFDC loan ಸಿಗುತ್ತದೆ.",
        "native_fallback": "ಕ್ಷಮಿಸಿ, ಈಗ ಉತ್ತರ ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಕೇಳಿ.",
        "roman_fallback": "Sorry, ivaga answer fetch agalilla. Please matte keli.",
    },
    "ml-IN": {
        "name": "Malayalam",
        "script_name": "Malayalam script (മലയാളം)",
        "teacher_dialect": "clear, polite, standard conversational Malayalam",
        "native_example": "PM-AJAY scheme-ൽ SC കുടുംബങ്ങൾക്ക് free skill training, asset subsidy, NSFDC loan ലഭ്യമാണ്.",
        "native_fallback": "ക്ഷമിക്കണം, ഇപ്പോൾ ഉത്തരം ലഭ്യമാക്കാൻ കഴിഞ്ഞില്ല. ദയവായി വീണ്ടും ചോദിക്കുക.",
        "roman_fallback": "Sorry, ippo answer fetch cheyyan pattiyilla. Please veendum chodikku.",
    },
    "mr-IN": {
        "name": "Marathi",
        "script_name": "Devanagari script (मराठी)",
        "teacher_dialect": "clear, polite, standard conversational Marathi",
        "native_example": "PM-AJAY scheme मध्ये SC कुटुंबांना free skill training, asset subsidy आणि NSFDC loan मिळतो.",
        "native_fallback": "क्षमस्व, आता उत्तर मिळू शकले नाही. कृपया पुन्हा विचारा.",
        "roman_fallback": "Sorry, ata answer fetch zala nahi. Please punha vichara.",
    },
    "en-IN": {
        "name": "English",
        "script_name": "English",
        "teacher_dialect": "clear, polite, standard Indian English",
        "native_example": "Under PM-AJAY, SC families with annual income below Rs 2.5 lakh get free skill training, asset subsidy up to Rs 50,000, and NSFDC loans.",
        "native_fallback": "Sorry, I could not fetch the answer. Please ask again.",
        "roman_fallback": "Sorry, I could not fetch the answer. Please ask again.",
    },
}

DEFAULT_PROFILE: dict[str, str] = LANGUAGE_PROFILES["en-IN"]
_LANGUAGE_PROFILES_BY_NORMALIZED_CODE = {
    code.lower(): profile for code, profile in LANGUAGE_PROFILES.items()
}

VERNACULAR_KEYWORD_PATTERNS = {
    "te-IN": re.compile(
        r"[\u0C00-\u0C7F]|\b(ante|enti|ela|mariyu|lo|unna|chese|cheppandi|cheyadam|kadha|gurinchi|enduku|chesukondi)\b",
        re.IGNORECASE,
    ),
    "ta-IN": re.compile(
        r"[\u0B80-\u0BFF]|\b(enna|epdi|adha|pannuvanga|solunga|la|kaaga|irukku|panlaam)\b",
        re.IGNORECASE,
    ),
    "bn-IN": re.compile(
        r"[\u0980-\u09FF]|\b(ki|bhabe|kaaj|kore|bolun|ekta|jekhane|kora|hoy)\b",
        re.IGNORECASE,
    ),
    "hi-IN": re.compile(
        r"[\u0900-\u097F]|\b(kya|kaise|hota|hoti|hai|bataiye|batao|karte|jisme|karega)\b",
        re.IGNORECASE,
    ),
}


def _is_native_script_mode() -> bool:
    settings = get_settings()
    return getattr(settings, "bhashini_tts_script_mode", "native").strip().lower() == "native"


def get_language_profile(language_code: str | None) -> dict[str, str]:
    """Return a case-insensitive caller-language profile, with a safe default."""
    normalized = str(language_code or "").strip().lower()
    raw_profile = _LANGUAGE_PROFILES_BY_NORMALIZED_CODE.get(normalized, DEFAULT_PROFILE)
    profile = raw_profile.copy()
    if _is_native_script_mode():
        profile["style"] = f"clean {profile.get('script_name', 'native script')} with scheme/finance terms in English"
        profile["example"] = profile.get("native_example", "")
        profile["fallback"] = profile.get("native_fallback", "")
    else:
        profile["style"] = profile.get("roman_fallback", "")
        profile["example"] = profile.get("native_example", "")
        profile["fallback"] = profile.get("roman_fallback", "")
    return profile


def _resolve_target_language(query: str, language_code: str = "en-IN") -> str:
    """
    Dynamically resolve target response language based on user's query and detected language.
    Mirrors the user's spoken language:
    - If user query has Telugu script / romanized -> te-IN
    - If user query has Hindi script / romanized -> hi-IN
    - If user query has Tamil script / romanized -> ta-IN
    - If user query has Kannada script / romanized -> kn-IN
    - If user query has Bengali script -> bn-IN
    - Otherwise honors requested language code or defaults to en-IN.
    """
    clean_q = query.strip()
    norm = str(language_code or "").strip().lower()

    # 1. Unambiguous native Unicode scripts in query ALWAYS override
    if re.search(r"[\u0C00-\u0C7F]", clean_q):  # Telugu script
        return "te-IN"
    if re.search(r"[\u0900-\u097F]", clean_q):  # Devanagari script
        return "hi-IN"
    if re.search(r"[\u0B80-\u0BFF]", clean_q):  # Tamil script
        return "ta-IN"
    if re.search(r"[\u0C80-\u0CFF]", clean_q):  # Kannada script
        return "kn-IN"
    if re.search(r"[\u0980-\u09FF]", clean_q):  # Bengali script
        return "bn-IN"

    # 2. Romanized Indic keywords override English default
    te_roman = re.compile(r"\b(ante|enti|ela|mariyu|unna|chese|cheppandi|cheyadam|kadha|gurinchi|enduku|chesukondi|kaavali|undi|telusukovalani|unnanu)\b", re.IGNORECASE)
    hi_roman = re.compile(r"\b(kya|kaise|hota|hoti|bataiye|batao|karte|karna|chahiye|mujhe|mera|meri|seekhna|chahata|chahati|padhai|karega)\b", re.IGNORECASE)
    ta_roman = re.compile(r"\b(enna|epdi|adha|pannuvanga|solunga|kaaga|irukku|panlaam|venum)\b", re.IGNORECASE)
    kn_roman = re.compile(r"\b(yenu|hege|beku|madabeku|thilisiri|nanna)\b", re.IGNORECASE)

    if te_roman.search(clean_q):
        return "te-IN"
    if hi_roman.search(clean_q):
        return "hi-IN"
    if ta_roman.search(clean_q):
        return "ta-IN"
    if kn_roman.search(clean_q):
        return "kn-IN"

    # 3. Explicit language code provided and recognized
    if norm not in {"", "unknown", "none"}:
        if norm.startswith("te"):
            return "te-IN"
        if norm.startswith("hi"):
            return "hi-IN"
        if norm.startswith("ta"):
            return "ta-IN"
        if norm.startswith("kn"):
            return "kn-IN"
        if norm.startswith("bn"):
            return "bn-IN"
        return "en-IN"

    # 4. Defaults to English (en-IN)
    return "en-IN"


# ─── System Prompt ────────────────────────────────────────────────────────────

PMAJAY_VOICE_SYSTEM_PROMPT = """
You are Kaushal Vaani (కౌశల్ వాణి / कौशल वाणी) - Virtual Livelihood Assistant — a warm, helpful AI career counsellor answering live telephone calls for Scheduled Caste (SC) beneficiaries and livelihood seekers.
You help callers understand PM-AJAY (Pradhan Mantri Anusuchit Jaati Abhyuday Yojana) scheme benefits, skill training opportunities, NSQF qualifications, asset subsidies, and credit linkages.

STRICT CRITICAL RULES:
1. Dynamic Language Mirroring:
   - Mirror the exact language of the caller's utterance.
   - If English: Respond entirely in clear, professional, empathetic Indian English. Explain PM-AJAY scheme, NSQF course, monthly salary/subsidy details, and center location. Do NOT use regional scripts.
   - If Telugu: Respond in natural Telugu script (తెలుగు). Keep technical terms in Latin script (PM-AJAY, NSQF, Solar PV Installer, NSFDC, Stipend, ITI, RSETI).
   - If Hindi: Respond in natural Hindi script (हिन्दी). Keep technical terms in Latin script (PM-AJAY, NSQF, Solar PV Installer, NSFDC, Stipend, ITI, RSETI).
   - If Tamil/Kannada: Respond in natural native script with technical terms in Latin script.

2. Brevity for Telephone:
   - Keep answers to 1-2 short spoken sentences (15-25 words maximum).
   - No bullet points, lists, markdown, or emoji — this is telephone speech.
   - Finish each sentence cleanly with a full stop.

3. Warm & Empathetic Tone:
   - Speak like a helpful government welfare counsellor guiding a rural beneficiary.
   - Be encouraging and use simple words. Avoid bureaucratic jargon.
   - End with an invitation to ask the next question.

4. Accuracy:
   - Use only facts from the provided context. Do NOT invent scheme amounts, rules, or center addresses.
   - If unsure, say honestly that you will connect them to the helpline: 1800-11-2001.
""".strip()


def _language_style_instruction(language_code: str) -> str:
    norm = str(language_code or "en-IN").strip().lower()
    if norm in ("en-in", "en"):
        return (
            "CALLER LANGUAGE: English (en-IN). "
            "Respond entirely in clear, professional, empathetic English. "
            "Explain the PM-AJAY scheme, NSQF training course, monthly salary or self-employment loan details, and training center location. "
            "Do NOT use Telugu, Hindi, or any regional script. "
            "Keep to 1-2 spoken sentences under 30 words."
        )
    elif norm in ("te-in", "te"):
        return (
            "CALLER LANGUAGE: Telugu (te-IN). "
            "Respond in natural Telugu script (తెలుగు). "
            "Keep technical terms, scheme titles, and numbers in Latin script: PM-AJAY, NSQF, Solar PV Installer, NSFDC, Stipend, ITI, RSETI. "
            "Keep to 1-2 spoken sentences under 30 words."
        )
    elif norm in ("hi-in", "hi"):
        return (
            "CALLER LANGUAGE: Hindi (hi-IN). "
            "Respond in natural Hindi script (हिन्दी). "
            "Keep technical terms, scheme titles, and numbers in Latin script: PM-AJAY, NSQF, Solar PV Installer, NSFDC, Stipend, ITI, RSETI. "
            "Keep to 1-2 spoken sentences under 30 words."
        )
    elif norm in ("ta-in", "ta"):
        return (
            "CALLER LANGUAGE: Tamil (ta-IN). "
            "Respond in natural Tamil script (தமிழ்). "
            "Keep technical terms and scheme titles in Latin script: PM-AJAY, NSQF, NSFDC, Stipend, ITI. "
            "Keep to 1-2 spoken sentences under 30 words."
        )
    elif norm in ("kn-in", "kn"):
        return (
            "CALLER LANGUAGE: Kannada (kn-IN). "
            "Respond in natural Kannada script (ಕನ್ನಡ). "
            "Keep technical terms and scheme titles in Latin script: PM-AJAY, NSQF, NSFDC, Stipend, ITI. "
            "Keep to 1-2 spoken sentences under 30 words."
        )
    else:
        profile = get_language_profile(language_code)
        return (
            f"CALLER LANGUAGE: {profile['name']} ({language_code}). "
            f"Respond in clean {profile.get('script_name', 'native script')} with technical terms in English Latin script. "
            "Keep to 1-2 spoken sentences under 30 words."
        )


def build_livelihood_recommendation(
    query: str,
    context_chunks: list[str],
    language: str = "en-IN",
) -> str:
    """
    Construct the full prompt string for the LLM based on retrieved context and caller language.
    Mirrors the caller's spoken language with tailored system instructions.
    """
    lang_instruction = _language_style_instruction(language)
    context_text = "\n\n".join(context_chunks) if context_chunks else "No specific context available."
    return (
        f"KNOWLEDGE BASE CONTEXT:\n{context_text}\n\n"
        f"CALLER QUESTION: {query}\n\n"
        f"RESPONSE LANGUAGE INSTRUCTION:\n{lang_instruction}"
    )


# ─── Groq client ──────────────────────────────────────────────────────────────

async def _get_groq_client() -> AsyncGroq:
    global _groq_client
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    async with _groq_client_lock:
        if _groq_client is None:
            _groq_client = AsyncGroq(api_key=settings.groq_api_key)
        return _groq_client


async def close_groq_client() -> None:
    global _groq_client
    async with _groq_client_lock:
        client, _groq_client = _groq_client, None
    if client is not None:
        await client.close()


# ─── Session ──────────────────────────────────────────────────────────────────

def _save_turn(session_id: str, question: str, answer: str) -> None:
    with _core_lock:
        session = _sessions.setdefault(session_id, [])
        session.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        del session[:-8]


def _session_history_copy(session_id: str) -> list[dict[str, str]]:
    with _core_lock:
        return _sessions.get(session_id, []).copy()


# ─── LLM generation ───────────────────────────────────────────────────────────

def _clean_spoken_text(text: str) -> str:
    """Normalize unicode quotes, dashes, and whitespace for TTS engines."""
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
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
        .strip()
    )


async def _generate_answer(
    query: str,
    context_chunks: list[str],
    session_id: str,
    language_code: str,
) -> str | None:
    """Call Groq with retrieved context to generate a spoken answer mirroring caller language."""
    settings = get_settings()
    if not settings.groq_api_key:
        logger.error("groq_not_configured")
        return None

    history = await asyncio.to_thread(_session_history_copy, session_id)
    user_content = build_livelihood_recommendation(query, context_chunks, language=language_code)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": PMAJAY_VOICE_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_content},
    ]

    try:
        client = await _get_groq_client()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.2,
                reasoning_effort="low",
                include_reasoning=False,
                max_tokens=120,
            ),
            timeout=settings.rag_query_timeout_seconds,
        )
        answer = response.choices[0].message.content
        if not isinstance(answer, str) or not answer.strip():
            return None
        answer = _clean_spoken_text(answer)
        await asyncio.to_thread(_save_turn, session_id, query, answer)
        return answer
    except Exception:
        logger.exception("groq_generation_failed")
        return None


# ─── Main public API ──────────────────────────────────────────────────────────

async def query_rag(
    user_query: str,
    session_id: str = "default_session",
    language_code: str = "en-IN",
    *,
    language: str | None = None,
) -> str | None:
    """
    Retrieve relevant Kaushal Vaani / PM-AJAY context with strict vocational trade filtering
    and dynamic language mirroring.
    Inspects user query and requested language, strictly mirroring the caller's spoken language.
    Returns None on failure (caller layer should play fallback audio).
    """
    from app.services.ingestion import query_vector_db, _detect_trade_intent

    target_lang = language or language_code or "en-IN"
    resolved_lang = _resolve_target_language(user_query, target_lang)

    # Detect trade intent from user query
    pref_sectors, boost_kws = _detect_trade_intent(user_query)

    try:
        context_chunks = await asyncio.wait_for(
            query_vector_db(
                user_query,
                n_results=5,
                preferred_sectors=pref_sectors,
                boost_keywords=boost_kws,
            ),
            timeout=4.0,
        )
    except Exception:
        context_chunks = []

    return await _generate_answer(user_query, context_chunks, session_id, resolved_lang)


async def warm_rag_index() -> None:
    """
    No-op warm-up kept for API compatibility.
    Actual initialization runs via auto_initialize_vector_db() in lifespan.
    """
    logger.info("rag_ready", extra={"event": "rag", "status": "pm_ajay_chroma"})
