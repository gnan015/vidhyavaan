import asyncio
import base64
import binascii
import io
import json
import logging
import os
import re
import struct
import time
import traceback
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.exotel import AudioFrameInfo, ExotelCallback
from app.services.audio import process_audio_frame
from app.services.recordings import download_and_process_recording, validate_recording_url
from app.services.sarvam import (
    sarvam_audio_to_pcm16,
    synthesize_speech_sarvam,
    transcribe_audio_file_sarvam,
    transcribe_audio_sarvam,
)
from app.services.rag_middleware import get_language_profile, query_rag
from app.services.security import verify_exotel_request

router = APIRouter(tags=["exotel"])
logger = logging.getLogger(__name__)
XML_MEDIA_TYPE = "application/xml"
STREAM_RECORDINGS_DIRECTORY = Path("recordings")
_transcription_tasks: set[asyncio.Task[dict[str, Any]]] = set()

# PM-AJAY fallback messages per language
_FALLBACK_RESPONSES = {
    "en-IN": "Sorry, I could not fetch the answer. Please ask again or call our helpline 1800-11-2001.",
    "hi-IN": "माफ़ कीजिए, अभी जानकारी नहीं मिली। कृपया फिर पूछें या helpline 1800-11-2001 पर call करें।",
    "te-IN": "క్షమించండి, ప్రస్తుతం సమాధానం దొరకలేదు. మళ్ళీ అడగండి లేదా helpline 1800-11-2001 కి call చేయండి.",
    "ta-IN": "மன்னிக்கவும், தகவல் கிடைக்கவில்லை. மீண்டும் கேளுங்கள் அல்லது helpline 1800-11-2001 call பண்ணுங்கள்.",
    "kn-IN": "ಕ್ಷಮಿಸಿ, ಮಾಹಿತಿ ಸಿಗಲಿಲ್ಲ. ಮತ್ತೆ ಕೇಳಿ ಅಥವಾ helpline 1800-11-2001 ಗೆ call ಮಾಡಿ.",
}

# ─── PM-AJAY Session State Machine ─────────────────────────────────────────────
# Each call goes through a structured 4-turn intake before free-form Q&A.
# turn 0: greeting + ask education level
# turn 1: ask family trade / current occupation
# turn 2: ask district
# turn 3: ask preference (wage vs self-employment)
# turn 4+: free-form PM-AJAY scheme Q&A using ChromaDB RAG
_CALL_SESSIONS: dict[str, dict] = {}

def _normalize_lang_code(code: str | None) -> str:
    if not code:
        return "en-IN"
    c = code.strip().lower()
    if c.startswith("te"):
        return "te-IN"
    if c.startswith("hi"):
        return "hi-IN"
    if c.startswith("ta"):
        return "ta-IN"
    if c.startswith("kn"):
        return "kn-IN"
    if c.startswith("bn"):
        return "bn-IN"
    return "en-IN"


def detect_transcript_language(text: str, asr_hint: str | None = None) -> str:
    """
    Dynamically detect the caller's spoken language from their transcript text and ASR hint.
    Returns standard language code: 'en-IN', 'te-IN', 'hi-IN', 'ta-IN', 'kn-IN', or 'bn-IN'.
    """
    clean_text = text.strip()
    if not clean_text:
        return _normalize_lang_code(asr_hint or "en-IN")

    # 1. Unambiguous native Unicode scripts
    if re.search(r"[\u0C00-\u0C7F]", clean_text):  # Telugu
        return "te-IN"
    if re.search(r"[\u0900-\u097F]", clean_text):  # Devanagari / Hindi
        return "hi-IN"
    if re.search(r"[\u0B80-\u0BFF]", clean_text):  # Tamil
        return "ta-IN"
    if re.search(r"[\u0C80-\u0CFF]", clean_text):  # Kannada
        return "kn-IN"
    if re.search(r"[\u0980-\u09FF]", clean_text):  # Bengali
        return "bn-IN"

    # 2. Romanized Indic keywords (when transliterated to Latin script)
    te_roman = re.compile(r"\b(ante|enti|ela|mariyu|unna|chese|cheppandi|cheyadam|kadha|gurinchi|enduku|chesukondi|kaavali|undi|telusukovalani|unnanu|uudyogam|chaduvukunna)\b", re.IGNORECASE)
    hi_roman = re.compile(r"\b(kya|kaise|hota|hoti|bataiye|batao|karte|karna|chahiye|mujhe|mera|meri|seekhna|chahata|chahati|padhai|karega)\b", re.IGNORECASE)
    ta_roman = re.compile(r"\b(enna|epdi|adha|pannuvanga|solunga|kaaga|irukku|panlaam|venum)\b", re.IGNORECASE)
    kn_roman = re.compile(r"\b(yenu|hege|beku|madabeku|thilisiri|nanna)\b", re.IGNORECASE)

    if te_roman.search(clean_text):
        return "te-IN"
    if hi_roman.search(clean_text):
        return "hi-IN"
    if ta_roman.search(clean_text):
        return "ta-IN"
    if kn_roman.search(clean_text):
        return "kn-IN"

    # 3. English lexical presence
    en_words = re.compile(
        r"\b(i|my|me|we|you|your|want|need|have|completed|pass|class|job|course|courses|training|salary|district|live|school|computer|computers|work|trade|experience|recommend|tell|about|what|which|where|how|can|is|are|in|at|to|for|with|and|the|wage|self|business|employment|help|hello|yes|no)\b",
        re.IGNORECASE,
    )
    en_matches = len(en_words.findall(clean_text))
    if en_matches >= 2:
        return "en-IN"

    # 4. If ASR provided a specific valid language hint
    if asr_hint and asr_hint.lower() not in ("unknown", "", "none"):
        return _normalize_lang_code(asr_hint)

    return "en-IN"


def _get_session(call_sid: str) -> dict:
    if call_sid not in _CALL_SESSIONS:
        _CALL_SESSIONS[call_sid] = {
            "turn": 0,
            "language": "en-IN",
            "caller_number": "",
            "profile": {
                "education": None,
                "family_trade": None,
                "district": None,
                "preference": None,
                "physical_constraint": None,
            },
        }
    return _CALL_SESSIONS[call_sid]


def _intake_question(turn: int, language: str) -> str:
    """Return the structured intake question for each turn in the caller's language."""
    questions: dict[int, dict[str, str]] = {
        0: {
            "en-IN": "Welcome to Kaushal Vaani - Virtual Livelihood Assistant. May I know your name and your highest education: primary school, 8th to 10th pass, or 12th pass?",
            "te-IN": "కౌశల్ వాణి (Kaushal Vaani) కి స్వాగతం. మీ పేరు మరియు మీ విద్యార్హత ఏమిటి — primary school, 8th to 10th pass, లేదా 12th pass?",
            "hi-IN": "कौशल वाणी (Kaushal Vaani) में आपका स्वागत है। आपका नाम और आपकी पढ़ाई कितनी है — primary school, 8th to 10th pass, या 12th pass?",
            "ta-IN": "கௌசல் வாணி (Kaushal Vaani)-க்கு வரவேற்கிறோம். உங்கள் பெயர் மற்றும் கல்வித் தகுதி என்ன — primary school, 8th to 10th pass, அல்லது 12th pass?",
            "kn-IN": "ಕೌಶಲ್ ವಾಣಿ (Kaushal Vaani) ಗೆ ಸ್ವಾಗತ. ನಿಮ್ಮ ಹೆಸರು ಮತ್ತು ವಿದ್ಯಾಭ್ಯಾಸ ಎಷ್ಟು — primary school, 8th to 10th pass, ಅಥವಾ 12th pass?",
        },
        1: {
            "en-IN": "Thank you. What work does your family currently do, or what is your trade experience, like farming, tailoring, construction, or artisan crafts?",
            "te-IN": "ధన్యవాదాలు. మీ కుటుంబ వృత్తి లేదా పని అనుభవం ఏమిటి — వ్యవసాయం (farming), tailoring, construction, లేదా artisan crafts?",
            "hi-IN": "धन्यवाद। आपका परिवार अभी क्या काम करता है, या आपका अनुभव क्या है — farming, tailoring, construction, या artisan crafts?",
            "ta-IN": "நன்றி. உங்கள் குடும்பத் தொழில் அல்லது பணி அனுபவம் என்ன — farming, tailoring, construction, அல்லது artisan crafts?",
            "kn-IN": "ಧನ್ಯವಾದ. ನಿಮ್ಮ ಕುಟುಂಬ ಈಗ ಏನು ಕೆಲಸ ಮಾಡುತ್ತಿದೆ, ಅಥವಾ ಕಸುಬು ಏನು — farming, tailoring, construction, ಅಥವಾ artisan crafts?",
        },
        2: {
            "en-IN": "Understood. Which district do you live in, and do you prefer a regular wage job or self-employment to start your own business?",
            "te-IN": "అర్థమైంది. మీరు ఏ district లో నివసిస్తున్నారు, మరియు మీకు regular ఉద్యోగం (wage job) కావాలా లేదా స్వయం ఉపాధి (self-employment) కావాలా?",
            "hi-IN": "समझ गया। आप किस district में रहते हैं, और क्या आप regular नौकरी चाहते हैं या अपना business (self-employment) शुरू करना चाहते हैं?",
            "ta-IN": "புரிந்தது. நீங்கள் எந்த district-ல் வசிக்கிறீர்கள், மற்றும் regular வேலை விரும்புகிறீர்களா அல்லது சுயதொழில் (self-employment) செய்ய விரும்புகிறீர்களா?",
            "kn-IN": "ತಿಳಿಯಿತು. ನೀವು ಯಾವ district-ನಲ್ಲಿ ವಾಸಿಸುತ್ತೀರಿ, ಮತ್ತು regular ಉದ್ಯೋಗ ಬೇಕೇ ಅಥವಾ ಸ್ವಂತ ಉದ್ಯಮ (self-employment) ಬೇಕೇ?",
        },
        3: {
            "en-IN": "Do you prefer regular wage employment, or starting your own small business or self-employment?",
            "te-IN": "మీకు regular job కావాలా, లేదా మీ సొంత business లేదా self-employment కావాలా?",
            "hi-IN": "क्या आप regular नौकरी चाहते हैं, या अपना खुद का छोटा business या self-employment?",
            "ta-IN": "நீங்கள் regular job விரும்புகிறீர்களா, அல்லது சொந்தமாக small business அல்லது self-employment?",
            "kn-IN": "ನಿಮಗೆ regular job ಬೇಕೇ, ಅಥವಾ ನಿಮ್ಮದೇ small business ಅಥವಾ self-employment ಬೇಕೇ?",
        },
    }
    lang_q = questions.get(turn, {})
    return lang_q.get(language, lang_q.get("en-IN", ""))


def _build_recommendation_query(profile: dict, language: str) -> str:
    """Build a PM-AJAY RAG query from the collected intake profile."""
    edu = profile.get("education") or "not specified"
    trade = profile.get("family_trade") or "not specified"
    district = profile.get("district") or "not specified"
    pref = profile.get("preference") or "not specified"
    constraint = profile.get("physical_constraint")

    query = (
        f"I am a Scheduled Caste beneficiary with education level {edu}, "
        f"my family's current work is {trade}, I live in {district} district, "
        f"and I prefer {pref}. "
        "What PM-AJAY scheme benefits, skill training trades, asset subsidy, and credit linkages "
        "are best suited for me? Suggest the most appropriate livelihood model."
    )
    if constraint:
        query += f" Note: I have a physical disability."
    return query


def exotel_hangup_xml() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'


def _stream_recording_path(call_sid: str | None) -> Path:
    """Return a traversal-safe filename for an Exotel stream recording."""
    safe_call_sid = re.sub(r"[^A-Za-z0-9_.-]", "_", call_sid or "unknown")
    return STREAM_RECORDINGS_DIRECTORY / f"call_{safe_call_sid}.wav"


def _fallback_response(language_code: str) -> str:
    """Return conversational recovery speech for the detected caller language."""
    return get_language_profile(language_code)["fallback"]


CENTERS_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "centers.json"
_CACHED_CENTERS: dict | None = None


def _resolve_nearest_center(district_query: str | None, preference: str | None = None) -> dict | None:
    """Resolve the nearest training center from centers.json by district and preference."""
    global _CACHED_CENTERS
    if _CACHED_CENTERS is None:
        if CENTERS_DATA_PATH.is_file():
            try:
                _CACHED_CENTERS = json.loads(CENTERS_DATA_PATH.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("centers_json_load_failed error=%s", exc)
                _CACHED_CENTERS = {}
        else:
            _CACHED_CENTERS = {}

    all_centers: list[dict] = []
    matched: list[dict] = []
    norm = (district_query or "").lower().strip()

    for state, dists in _CACHED_CENTERS.items():
        if isinstance(dists, dict):
            for dist, c_list in dists.items():
                if isinstance(c_list, list):
                    for c in c_list:
                        c_info = dict(c, state=state, district=dist)
                        all_centers.append(c_info)
                        if dist.lower() in norm or any(w in dist.lower() for w in norm.split() if len(w) > 3):
                            matched.append(c_info)

    candidates = matched or all_centers
    is_self = "self" in (preference or "").lower() or "business" in (preference or "").lower()
    for c in candidates:
        c_type = c.get("type", "").lower()
        if is_self and ("rseti" in c_type or "rudset" in c_type):
            return c
        elif not is_self and "iti" in c_type:
            return c
    return candidates[0] if candidates else None


async def send_sms_followup(caller_number: str, message: str) -> bool:
    """Send automated follow-up SMS via Exotel SMS API or log for dispatch."""
    import os
    import httpx
    clean_number = re.sub(r"[^0-9+]", "", caller_number or "")
    if not clean_number or len(clean_number) < 10:
        logger.info("[SMS FOLLOWUP] Skipped: Invalid or missing phone number '%s'.", caller_number)
        return False

    settings = get_settings()
    username = getattr(settings, "exotel_username", None) or os.getenv("EXOTEL_USERNAME", "")
    password = getattr(settings, "exotel_password", None) or os.getenv("EXOTEL_PASSWORD", "")
    account_sid = getattr(settings, "exotel_account_sid", None) or os.getenv("EXOTEL_ACCOUNT_SID", "") or username
    subdomain = getattr(settings, "exotel_subdomain", None) or os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")

    logger.info("[SMS FOLLOWUP] Dispatching to %s: %s", clean_number, message[:80])
    print(f"[SMS FOLLOWUP] Dispatching to {clean_number}: {message}", flush=True)

    if username and password and account_sid:
        url = f"https://{subdomain}/v1/Accounts/{account_sid}/Sms/send.json"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    auth=(username, password),
                    data={
                        "From": getattr(settings, "exotel_caller_id", None) or os.getenv("EXOTEL_CALLER_ID", "08088919888"),
                        "To": clean_number,
                        "Body": message,
                    },
                )
                logger.info("[SMS FOLLOWUP] Exotel SMS response code=%d", resp.status_code)
                return resp.status_code in (200, 201)
        except Exception as exc:
            logger.warning("[SMS FOLLOWUP] Exotel SMS dispatch failed: %s", exc)
            return False
    return True


def _trim_trailing_silence(audio: bytes, sample_rate: int, keep_silence_ms: int = 120) -> bytes:
    """Trim silence frames at the end of an utterance to speed up STT upload and inference."""
    frame_size = sample_rate * 2 // 10  # 100ms
    if len(audio) <= frame_size * 2:
        return audio
    keep_bytes = (sample_rate * 2 * keep_silence_ms) // 1000
    threshold = 300
    last_sound_idx = len(audio)
    for idx in range(len(audio) - frame_size, 0, -frame_size):
        chunk = audio[idx : idx + frame_size]
        if _pcm_rms(chunk) >= threshold:
            last_sound_idx = min(len(audio), idx + frame_size + keep_bytes)
            break
    return audio[:last_sound_idx]


def _sample_rate_from_media_format(media_format: object) -> int:
    """Read Exotel's media format variants, falling back to Exotel's 8 kHz default."""
    if not isinstance(media_format, dict):
        return 8_000
    candidate = (
        media_format.get("sample_rate")
        or media_format.get("sampleRate")
        or media_format.get("rate")
    )
    try:
        rate = int(candidate)
    except (TypeError, ValueError):
        return 8_000
    return rate if rate > 0 else 8_000


def _sample_width_from_media_format(media_format: dict[str, Any]) -> int:
    """Determine the PCM sample width from Exotel's bit-rate metadata."""
    bit_rate = media_format.get("bit_rate", media_format.get("bitRate", "16"))
    try:
        bits = int(bit_rate)
    except (TypeError, ValueError):
        bits = 16
    return 2 if bits == 16 else max(1, (bits + 7) // 8)


def _ulaw_to_pcm16(chunk: bytes) -> bytes:
    """Decode G.711 mu-law to signed, little-endian 16-bit linear PCM.

    This small decoder avoids the deprecated/removed ``audioop`` dependency and
    works on Python 3.13+ as well as current Python releases.
    """
    samples = bytearray(len(chunk) * 2)
    for index, value in enumerate(chunk):
        ulaw = ~value & 0xFF
        magnitude = ((ulaw & 0x0F) << 3) + 0x84
        magnitude <<= (ulaw & 0x70) >> 4
        sample = (0x84 - magnitude) if (ulaw & 0x80) else (magnitude - 0x84)
        struct.pack_into("<h", samples, index * 2, sample)
    return bytes(samples)


def _write_stream_wav(
    audio: bytes, call_sid: str | None, sample_rate: int, sample_width: int
) -> Path:
    """Persist little-endian 16-bit mono PCM as a standards-compliant WAV file."""
    os.makedirs(STREAM_RECORDINGS_DIRECTORY, exist_ok=True)
    output_path = _stream_recording_path(call_sid)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)
    return output_path


def _schedule_transcription(audio_path: Path, call_sid: str | None) -> None:
    """Keep the post-call network task alive without delaying socket teardown."""
    task = asyncio.create_task(
        transcribe_audio_file_sarvam(str(audio_path)), name="sarvam-transcription"
    )
    _transcription_tasks.add(task)
    task.add_done_callback(_transcription_tasks.discard)
    logger.info(
        "sarvam_transcription_scheduled",
        extra={"call_sid": call_sid, "recording_path": str(audio_path), "event": "transcription"},
    )


def _smooth_and_normalize_pcm(audio: bytes, sample_rate: int, sample_width: int) -> bytes:
    """Remove DC bias, apply 50 ms edge fades, and normalize PCM16 safely."""
    if sample_width != 2 or len(audio) < 2:
        return audio
    usable_length = len(audio) - (len(audio) % 2)
    samples = list(struct.unpack(f"<{usable_length // 2}h", audio[:usable_length]))
    if not samples:
        return b""

    dc_offset = sum(samples) / len(samples)
    centered = [sample - dc_offset for sample in samples]
    rms = (sum(sample * sample for sample in centered) / len(centered)) ** 0.5
    peak = max((abs(sample) for sample in centered), default=0)
    target_rms = 0.20 * 32767  # Speech-friendly target with headroom.
    gain = target_rms / rms if rms else 1.0
    if peak:
        gain = min(gain, (0.95 * 32767) / peak)

    fade_samples = min(int(sample_rate * 0.050), len(centered) // 2)
    output: list[int] = []
    for index, sample in enumerate(centered):
        fade = 1.0
        if fade_samples:
            if index < fade_samples:
                fade = index / fade_samples
            elif index >= len(centered) - fade_samples:
                fade = (len(centered) - 1 - index) / fade_samples
        output.append(max(-32768, min(32767, round(sample * gain * fade))))
    return struct.pack(f"<{len(output)}h", *output)


def _pcm_rms(audio: bytes) -> float:
    """Return RMS energy for a little-endian PCM16 chunk without audioop."""
    usable_length = len(audio) - (len(audio) % 2)
    if not usable_length:
        return 0.0
    samples = struct.iter_unpack("<h", audio[:usable_length])
    squared_sum = 0
    count = 0
    for (sample,) in samples:
        squared_sum += sample * sample
        count += 1
    return (squared_sum / count) ** 0.5 if count else 0.0


def _pcm_to_wav_bytes(audio: bytes, sample_rate: int) -> bytes:
    """Wrap Exotel PCM16 in a WAV container for Sarvam inference."""
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)
    return stream.getvalue()


def _turn_telemetry(message: str, *, call_sid: str | None = None) -> None:
    """Print concise turn telemetry and retain it in structured application logs."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        safe_msg = message.encode("ascii", "replace").decode("ascii")
        print(safe_msg, flush=True)
    logger.info(message, extra={"call_sid": call_sid, "event": "loopback"})


def _stage_error(stage: str, error: Exception, *, call_sid: str | None) -> None:
    """Log a recoverable turn-stage failure without ending the WebSocket call."""
    message = f"[ERROR AT STAGE: {stage}] {type(error).__name__}: {error}"
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)
    logger.exception(
        "loopback_stage_failed",
        extra={"call_sid": call_sid, "event": "loopback", "stage": stage},
    )
    traceback.print_exc()


class _ExotelOutboundSender:
    """Serialize outbound media so keepalive and TTS frames cannot interleave."""

    def __init__(self, websocket: WebSocket, stream_sid: str) -> None:
        self._websocket = websocket
        self._stream_sid = stream_sid
        self._lock = asyncio.Lock()
        self._closed = False

    def close(self) -> None:
        self._closed = True

    async def send_pcm(self, audio: bytes, sample_rate: int) -> None:
        """Send Exotel's minimal documented bidirectional media envelope."""
        if self._closed:
            return
        async with self._lock:
            if self._closed:
                return
            try:
                if getattr(self._websocket, "client_state", None) != WebSocketState.CONNECTED:
                    self._closed = True
                    return
                await self._websocket.send_json(
                    {
                        "event": "media",
                        "stream_sid": self._stream_sid,
                        "media": {
                            "payload": base64.b64encode(audio).decode("ascii"),
                        },
                    }
                )
            except (RuntimeError, WebSocketDisconnect, Exception) as exc:
                self._closed = True
                logger.debug("outbound_send_skipped error=%s", exc)

DEFAULT_TELEPHONY_FRAME_MS = 20  # Strict 20ms frames for Exotel telephony streaming


def _exotel_frame_size(sample_rate: int, duration_ms: int = DEFAULT_TELEPHONY_FRAME_MS) -> int:
    """Return one 20ms frame of mono 16-bit PCM for Exotel playback (320 bytes @ 8kHz, 640 bytes @ 16kHz)."""
    return int((sample_rate * 2 * duration_ms) // 1000)


async def _stream_pcm_to_exotel(
    websocket: WebSocket,
    stream_sid: str,
    audio: bytes,
    sample_rate: int,
    sender: _ExotelOutboundSender | None = None,
    barge_in_event: asyncio.Event | None = None,
    frame_duration_ms: int = DEFAULT_TELEPHONY_FRAME_MS,
) -> int:
    """Send Exotel-compliant PCM16 frames at real-time pace with barge-in support.

    Synthesized audio is broken into strict 20ms frames (e.g., 320 bytes at 8 kHz or 640 bytes at 16 kHz)
    and paced using drift-free monotonic clock timing to prevent buffer underruns, jitter, and stuttering.
    """
    frame_size = _exotel_frame_size(sample_rate, frame_duration_ms)
    frame_delay = frame_duration_ms / 1000.0
    frames_sent = 0
    sender = sender or _ExotelOutboundSender(websocket, stream_sid)
    start_time = time.monotonic()

    for offset in range(0, len(audio), frame_size):
        if barge_in_event and barge_in_event.is_set():
            logger.info("playback_interrupted_by_barge_in", extra={"stream_sid": stream_sid})
            break
        chunk = audio[offset : offset + frame_size]
        # Pad short final frame with PCM silence (0x00) to maintain uniform frame size
        if len(chunk) < frame_size:
            chunk = chunk.ljust(frame_size, b"\x00")
        await sender.send_pcm(chunk, sample_rate)
        frames_sent += 1

        # Drift-free real-time frame pacing
        expected_elapsed = frames_sent * frame_delay
        actual_elapsed = time.monotonic() - start_time
        sleep_needed = expected_elapsed - actual_elapsed
        if sleep_needed > 0:
            await asyncio.sleep(sleep_needed)
        else:
            await asyncio.sleep(0.001)

    return frames_sent


async def _stream_processing_keepalive(
    sender: _ExotelOutboundSender,
    sample_rate: int,
    stop_event: asyncio.Event,
    call_sid: str | None,
    language_code: str = "te-IN",
    *,
    first_frame_already_sent: bool = False,
) -> None:
    """Keep the Voicebot stream active while STT/RAG/TTS processes.
    Streams silent frames (no holding speech or filler audio) at 100ms intervals."""
    frame_size = _exotel_frame_size(sample_rate, DEFAULT_TELEPHONY_FRAME_MS)
    silence_frame = b"\x00" * frame_size
    sent = 0
    try:
        interval_seconds = 0.2
        while not stop_event.is_set():
            await sender.send_pcm(silence_frame, sample_rate)
            sent += 1
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except TimeoutError:
                pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _stage_error("KEEPALIVE", exc, call_sid=call_sid)


async def _stream_idle_keepalive(
    sender: _ExotelOutboundSender,
    sample_rate: int,
    stop_event: asyncio.Event,
    call_sid: str | None,
) -> None:
    """Keep a newly connected Voicebot call alive while waiting for caller speech."""
    silence_frame = b"\x00" * _exotel_frame_size(sample_rate)
    sent = 0
    _turn_telemetry(
        "[KEEPALIVE] Idle stream keepalive started; waiting for caller speech.",
        call_sid=call_sid,
    )
    try:
        while not stop_event.is_set():
            await sender.send_pcm(silence_frame, sample_rate)
            sent += 1
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.2)
            except TimeoutError:
                pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _stage_error("IDLE_KEEPALIVE", exc, call_sid=call_sid)
    finally:
        _turn_telemetry(
            f"[KEEPALIVE] Idle stream keepalive stopped after {sent} frame(s).",
            call_sid=call_sid,
        )


async def callback_payload(request: Request) -> dict[str, Any]:
    if request.method == "GET":
        return dict(request.query_params)
    content_type = request.headers.get("content-type", "").lower()
    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        return dict(await request.form())
    if "application/json" in content_type:
        content = await request.json()
        if isinstance(content, dict):
            return content
    raise HTTPException(status_code=415, detail="Expected form-encoded or JSON payload")


@router.api_route("/api/exotel/callback", methods=["GET", "POST"])
async def exotel_callback(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> Response:
    await verify_exotel_request(request, settings)
    try:
        callback = ExotelCallback.model_validate(await callback_payload(request))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    recording_url = str(callback.recording_url) if callback.recording_url else None
    if recording_url:
        validate_recording_url(recording_url, settings)
        background_tasks.add_task(download_and_process_recording, callback.call_sid, recording_url, settings)
    logger.info("exotel_callback_received", extra={"call_sid": callback.call_sid, "recording_url": recording_url, "event": "callback"})
    # Switch this to Gather/Dial XML if the selected Exotel applet expects continuation.
    return Response(content=exotel_hangup_xml(), media_type=XML_MEDIA_TYPE)


@router.websocket("/ws/exotel-stream")
async def exotel_stream(websocket: WebSocket) -> None:
    settings = get_settings()
    token = websocket.headers.get("X-Exotel-Token") or websocket.query_params.get("token")
    if settings.exotel_webhook_token and token != settings.exotel_webhook_token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[EXOTEL WS CONNECTED] Client IP: {client_host}", flush=True)
    call_sid: str | None = None
    stream_sid: str | None = None
    sample_rate = 8_000
    sample_width = 2
    encoding = "audio/x-raw"
    audio_frames = bytearray()
    audio_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=100)
    media_chunk_count = 0
    frames_received = 0
    frames_processed = 0
    sequence_gaps = 0
    last_sequence: int | None = None
    last_chunk_size = 0
    barge_in_event = asyncio.Event()
    speech_active = False
    last_speech_time: float | None = None
    is_bot_turn = False
    active_caller_language: str = settings.default_caller_language or "en-IN"
    utterance_task: asyncio.Task[None] | None = None
    outbound_sender: _ExotelOutboundSender | None = None
    idle_keepalive_stop = asyncio.Event()
    idle_keepalive_task: asyncio.Task[None] | None = None

    async def stop_idle_keepalive() -> None:
        """Stop call-wait keepalive before TTS playback uses the same sender."""
        nonlocal idle_keepalive_task
        idle_keepalive_stop.set()
        if idle_keepalive_task and not idle_keepalive_task.done():
            with suppress(asyncio.CancelledError):
                await idle_keepalive_task
        idle_keepalive_task = None

    async def start_idle_keepalive() -> None:
        """Start immediately after the Exotel stream ID is known, once only."""
        nonlocal idle_keepalive_task
        if not outbound_sender or not stream_sid:
            return
        if getattr(websocket, "client_state", None) != WebSocketState.CONNECTED:
            return
        if idle_keepalive_task and not idle_keepalive_task.done():
            return
        idle_keepalive_stop.clear()
        try:
            await outbound_sender.send_pcm(b"\x00" * _exotel_frame_size(sample_rate), sample_rate)
        except Exception:
            return
        idle_keepalive_task = asyncio.create_task(
            _stream_idle_keepalive(outbound_sender, sample_rate, idle_keepalive_stop, call_sid),
            name="exotel-idle-keepalive",
        )

    async def process_utterance(captured_audio: bytes) -> None:
        """Run one recoverable STT -> 4-turn intake / RAG -> TTS turn."""
        nonlocal speech_active, last_speech_time, is_bot_turn, utterance_task, active_caller_language
        turn_started = time.perf_counter()
        playback_sample_rate = settings.exotel_playback_sample_rate or sample_rate
        keepalive_stop = asyncio.Event()
        keepalive_task: asyncio.Task[None] | None = None

        # Retrieve or create session for this call
        session_id = call_sid or stream_sid or "anonymous"
        session = _get_session(session_id)
        session["language"] = active_caller_language

        async def stop_keepalive() -> None:
            keepalive_stop.set()
            if keepalive_task and not keepalive_task.done():
                with suppress(asyncio.CancelledError):
                    await keepalive_task

        async def play_failure_message(language_code: str = "en-IN") -> None:
            """Speak a fixed recovery message while preserving the active call."""
            if not stream_sid or not outbound_sender:
                return
            try:
                # This is already written in the caller's spoken style. Do not
                # mechanically translate it, because that produces unnatural
                # technical vocabulary in the live phone response.
                fallback_text = _fallback_response(language_code)
                _turn_telemetry(
                    "[TTS START] Synthesizing the recovery response via Sarvam...",
                    call_sid=call_sid,
                )
                fallback_started = time.perf_counter()
                fallback_wav = await synthesize_speech_sarvam(
                    fallback_text,
                    target_language_code=language_code,
                    speech_sample_rate=playback_sample_rate if playback_sample_rate > 0 else 8000,
                )
                fallback_pcm = sarvam_audio_to_pcm16(
                    fallback_wav, playback_sample_rate
                )
                _turn_telemetry(
                    "[TTS FINISH] Generated recovery audio in "
                    f"{time.perf_counter() - fallback_started:.2f}s",
                    call_sid=call_sid,
                )
                await stop_keepalive()
                frames = (len(fallback_pcm) + _exotel_frame_size(playback_sample_rate) - 1) // _exotel_frame_size(playback_sample_rate)
                _turn_telemetry(
                    f"[PLAYBACK] Streaming {frames} recovery frame(s) back to Exotel...",
                    call_sid=call_sid,
                )
                _turn_telemetry(
                    "[TOTAL TURNAROUND] "
                    f"{time.perf_counter() - turn_started:.2f}s to recovery playback frame.",
                    call_sid=call_sid,
                )
                await _stream_pcm_to_exotel(
                    websocket,
                    stream_sid,
                    fallback_pcm,
                    playback_sample_rate,
                    outbound_sender,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _stage_error("FALLBACK_TTS", exc, call_sid=call_sid)

        try:
            if not stream_sid or not outbound_sender:
                logger.warning(
                    "loopback_skipped_without_stream_sid",
                    extra={"call_sid": call_sid, "event": "loopback"},
                )
                return
            # The call-level keepalive runs while waiting for caller speech.
            # Stop it before the processing keepalive/TTS sequence takes over.
            await stop_idle_keepalive()
            # Send this frame before starting any external HTTP work.  Creating
            # a background task alone is insufficient if DNS/TLS setup delays
            # the event loop before that task gets a chance to run.
            try:
                await outbound_sender.send_pcm(
                    b"\x00" * _exotel_frame_size(playback_sample_rate),
                    playback_sample_rate,
                )
                _turn_telemetry(
                    "[KEEPALIVE] Initial silent frame sent before STT.",
                    call_sid=call_sid,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _stage_error("KEEPALIVE", exc, call_sid=call_sid)
                return
            keepalive_task = asyncio.create_task(
                _stream_processing_keepalive(
                    outbound_sender,
                    playback_sample_rate,
                    keepalive_stop,
                    call_sid,
                    language_code=active_caller_language,
                    first_frame_already_sent=True,
                ),
                name="exotel-processing-keepalive",
            )
            # Give the keepalive task one scheduling turn before the STT worker
            # starts its network request.
            await asyncio.sleep(0)

            try:
                trimmed_audio = _trim_trailing_silence(captured_audio, sample_rate)
                wav_audio = _pcm_to_wav_bytes(trimmed_audio, sample_rate)
                _turn_telemetry(
                    f"[STT START] Sending {len(wav_audio):,} bytes (trimmed from {len(captured_audio):,}) to Sarvam STT...",
                    call_sid=call_sid,
                )
                stt_started = time.perf_counter()
                recognition = await transcribe_audio_sarvam(
                    wav_audio,
                    language_code="unknown",
                )
                user_query = str(recognition.get("transcript") or recognition.get("english_query") or "").strip()
                asr_lang = recognition.get("language_code") or recognition.get("detected_language_code")
                detected_language_code = detect_transcript_language(user_query, asr_lang)
                session["language"] = detected_language_code
                active_caller_language = detected_language_code
                print(f"[LANGUAGE DETECTED & MIRRORED]: {detected_language_code} | Query: {user_query}", flush=True)
                _turn_telemetry(
                    "[STT FINISH] Transcribed in "
                    f"{time.perf_counter() - stt_started:.2f}s | Language: "
                    f"'{detected_language_code}' | Query: '{user_query[:500]}'",
                    call_sid=call_sid,
                )
                if not user_query:
                    _turn_telemetry(
                        "[STT] Empty/noise-only utterance. Listening again.",
                        call_sid=call_sid,
                    )
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _stage_error("STT", exc, call_sid=call_sid)
                await play_failure_message(session.get("language", active_caller_language))
                return

            # ── 4-Turn Structured Intake State Machine ───────────────────────
            current_turn = session.get("turn", 0)
            profile = session.setdefault("profile", {})
            call_lang = session.get("language", active_caller_language)

            if current_turn == 0:
                # Turn 0 reply = caller's name & education level
                profile["education"] = user_query
                name_match = re.search(r"(?:my name is|i am|name is|పేరు|నా పేరు|मेरा नाम|नाम)\s+([A-Za-z\u0C00-\u0C7F\u0900-\u097F]+)", user_query, re.IGNORECASE)
                if name_match:
                    profile["name"] = name_match.group(1).strip()
                session["turn"] = 1
                english_answer = _intake_question(1, call_lang)
                _turn_telemetry(f"[INTAKE T1] education='{user_query}' name='{profile.get('name')}' lang='{call_lang}'", call_sid=call_sid)

            elif current_turn == 1:
                # Turn 1 reply = family trade / occupation experience
                profile["family_trade"] = user_query
                session["turn"] = 2
                english_answer = _intake_question(2, call_lang)
                _turn_telemetry(f"[INTAKE T2] trade='{user_query}' lang='{call_lang}'", call_sid=call_sid)

            elif current_turn in (2, 3):
                # Turn 2 reply = district & employment preference (wage vs self-employed)
                resp_lower = user_query.lower()
                if any(kw in resp_lower for kw in ["self", "business", "own", "swa", "apna", "swayam", "udyam", "entrepren", "mushroom", "poultry", "dairy", "tailor", "shop", "vyapar"]):
                    profile["preference"] = "self-employment"
                elif any(kw in resp_lower for kw in ["wage", "job", "naukri", "regular", "company", "salary"]):
                    profile["preference"] = "wage employment"
                elif profile.get("preference") is None:
                    profile["preference"] = "wage employment"

                if profile.get("district") is None:
                    profile["district"] = user_query

                # Advance to Turn 3 (final recommendation + automated SMS)
                session["turn"] = 3
                _turn_telemetry(f"[INTAKE T3] district='{profile['district']}' preference='{profile['preference']}' lang='{call_lang}'", call_sid=call_sid)

                # Instantly query vector store and resolve nearest center
                nearest_center = _resolve_nearest_center(profile.get("district"), profile.get("preference"))
                center_name = nearest_center.get("center_name", "District Skill Training Center") if nearest_center else "District Skill Training Center"
                center_phone = nearest_center.get("phone", "1800-11-2001") if nearest_center else "1800-11-2001"
                center_dist = nearest_center.get("district", "your district") if nearest_center else "your district"

                rag_query = _build_recommendation_query(profile, call_lang)
                if settings.live_rag_enabled:
                    try:
                        _turn_telemetry("[RAG START] Personalised scheme recommendation...", call_sid=call_sid)
                        rag_started = time.perf_counter()
                        english_answer = await query_rag(rag_query, session_id, language=call_lang)
                        if not english_answer:
                            raise RuntimeError("RAG returned no answer for personalised query")
                        _turn_telemetry(
                            f"[RAG FINISH] {time.perf_counter() - rag_started:.2f}s | '{english_answer[:300]}'",
                            call_sid=call_sid,
                        )
                    except Exception as exc:
                        _stage_error("RAG", exc, call_sid=call_sid)
                        english_answer = (
                            f"Based on your profile in {center_dist}, you are eligible for free PM-AJAY skill training and asset subsidy up to Rs 50,000. "
                            f"Your nearest center is {center_name}. You can also access NSFDC concessional loans."
                        )
                else:
                    english_answer = (
                        f"Based on your profile in {center_dist}, you are eligible for free PM-AJAY skill training and asset subsidy up to Rs 50,000. "
                        f"Your nearest center is {center_name}. You can also access NSFDC concessional loans."
                    )

                # Trigger automated SMS follow-up
                caller_num = session.get("caller_number") or ""
                caller_name = profile.get("name") or "Beneficiary"
                sms_text = (
                    f"Kaushal Vaani (PM-AJAY) Recommendation: Dear {caller_name}, your career plan is ready. "
                    f"Benefits: Free PM-AJAY Skill Training, Asset Subsidy up to Rs 50,000, NSFDC Loans. "
                    f"Nearest Center: {center_name} ({center_phone}). Helpline: 1800-11-2001."
                )
                if caller_num:
                    asyncio.create_task(send_sms_followup(caller_num, sms_text))

                # Advance to 4 so caller can ask follow-up questions
                session["turn"] = 4

            else:
                # Turn 4+: free-form PM-AJAY scheme Q&A
                if settings.live_rag_enabled:
                    try:
                        _turn_telemetry(
                            "[RAG START] Querying PM-AJAY knowledge base...", call_sid=call_sid
                        )
                        rag_started = time.perf_counter()
                        english_answer = await query_rag(
                            user_query,
                            session_id,
                            language=call_lang,
                        )
                        if not english_answer:
                            raise RuntimeError("RAG middleware returned no usable answer")
                        _turn_telemetry(
                            f"[RAG FINISH] {time.perf_counter() - rag_started:.2f}s | '{english_answer[:300]}'",
                            call_sid=call_sid,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        _stage_error("RAG", exc, call_sid=call_sid)
                        await play_failure_message(call_lang)
                        return
                else:
                    english_answer = user_query
                    _turn_telemetry("[RAG SKIPPED] Direct response mode is active.", call_sid=call_sid)


            english_answer = (
                english_answer
                .replace("\u2011", "-")
                .replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("\u201c", '"')
                .replace("\u201d", '"')
            )

            # RAG/Groq owns both brevity and sentence completion. Do not slice
            # generated text in Python: that can cut a spoken answer mid-sentence.
            spoken_answer = english_answer.strip()
            if len(spoken_answer) > 1000:
                spoken_answer = spoken_answer[:1000]

            try:
                _turn_telemetry(
                    f"[TTS INPUT] {len(spoken_answer)} characters in direct voice style.",
                    call_sid=call_sid,
                )
                _turn_telemetry(
                    "[TTS START] Synthesizing speech via Sarvam...", call_sid=call_sid
                )
                tts_started = time.perf_counter()
                tts_wav = await synthesize_speech_sarvam(
                    spoken_answer,
                    target_language_code=call_lang,
                    speech_sample_rate=playback_sample_rate if playback_sample_rate > 0 else 8000,
                )
                response_pcm = sarvam_audio_to_pcm16(tts_wav, playback_sample_rate)
                if not response_pcm:
                    raise ValueError("Sarvam TTS returned no playable PCM audio")
                _turn_telemetry(
                    "[TTS FINISH] Generated "
                    f"{playback_sample_rate // 1000}kHz PCM audio in "
                    f"{time.perf_counter() - tts_started:.2f}s",
                    call_sid=call_sid,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _stage_error("TTS", exc, call_sid=call_sid)
                await play_failure_message(call_lang)
                return

            try:
                await stop_keepalive()
                if barge_in_event and barge_in_event.is_set():
                    _turn_telemetry("[PLAYBACK SKIPPED] Caller barged in before playback started.", call_sid=call_sid)
                    return
                frame_size = _exotel_frame_size(playback_sample_rate)
                expected_frames = (len(response_pcm) + frame_size - 1) // frame_size
                _turn_telemetry(
                    f"[PLAYBACK] Streaming {expected_frames} frame(s) back to Exotel...",
                    call_sid=call_sid,
                )
                playback_started = time.perf_counter()
                _turn_telemetry(
                    "[TOTAL TURNAROUND] "
                    f"{playback_started - turn_started:.2f}s to first playback frame.",
                    call_sid=call_sid,
                )
                frames_sent = await _stream_pcm_to_exotel(
                    websocket,
                    stream_sid,
                    response_pcm,
                    playback_sample_rate,
                    outbound_sender,
                    barge_in_event=barge_in_event,
                )
                _turn_telemetry(
                    "[READY] Playback complete. Listening for next turn. "
                    f"({frames_sent} frame(s), {time.perf_counter() - playback_started:.2f}s)",
                    call_sid=call_sid,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _stage_error("PLAYBACK", exc, call_sid=call_sid)
                # The receive loop remains open; the caller can try another turn.
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _stage_error("TURN", exc, call_sid=call_sid)
        finally:
            await stop_keepalive()
            _turn_telemetry(
                "[TURN COMPLETE] "
                f"{time.perf_counter() - turn_started:.2f}s",
                call_sid=call_sid,
            )
            audio_frames.clear()
            speech_active = False
            last_speech_time = None
            is_bot_turn = False
            utterance_task = None
            try:
                await start_idle_keepalive()
            except Exception as exc:
                _stage_error("IDLE_KEEPALIVE", exc, call_sid=call_sid)

    async def consume_audio() -> None:
        """Decode queued frames without delaying WebSocket reads."""
        nonlocal frames_processed, sequence_gaps, last_sequence, last_chunk_size
        nonlocal speech_active, last_speech_time, is_bot_turn, utterance_task
        while True:
            queued_packet = await audio_queue.get()
            try:
                if queued_packet is None:
                    return
                payload = queued_packet["payload"]
                try:
                    audio = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    logger.warning(
                        "invalid_media_payload",
                        extra={"call_sid": call_sid, "event": "media"},
                    )
                    continue
                if encoding == "audio/x-mulaw":
                    audio = _ulaw_to_pcm16(audio)

                sequence = queued_packet.get("sequence")
                if sequence is not None:
                    try:
                        sequence = int(sequence)
                    except (TypeError, ValueError):
                        sequence = None
                if sequence is not None and last_sequence is not None:
                    if sequence <= last_sequence:
                        logger.info(
                            "duplicate_or_out_of_order_media_ignored",
                            extra={"call_sid": call_sid, "event": "media"},
                        )
                        continue
                    missing_frames = sequence - last_sequence - 1
                    if missing_frames:
                        # Exotel WebSocket packets are ordered; a jump represents
                        # caller audio intentionally ignored during bot playback.
                        sequence_gaps += missing_frames
                        logger.warning(
                            "media_sequence_gap_detected",
                            extra={"call_sid": call_sid, "event": "media", "sequence_gaps": sequence_gaps},
                        )
                if sequence is not None:
                    last_sequence = sequence
                last_chunk_size = len(audio)
                frames_processed += 1
                if frames_processed % 50 == 0:
                    logger.info(
                        "exotel_media_processed",
                        extra={"call_sid": call_sid, "event": "media", "chunk_count": frames_processed},
                    )
                if sample_width == 2:
                    try:
                        now = time.monotonic()
                        energy = _pcm_rms(audio)
                        if is_bot_turn:
                            # Natural barge-in: If caller starts speaking while bot is talking
                            if energy >= settings.vad_rms_threshold * 1.5:
                                _turn_telemetry("[BARGE-IN] Caller interrupted bot playback.", call_sid=call_sid)
                                barge_in_event.set()
                                if utterance_task and not utterance_task.done():
                                    utterance_task.cancel()
                                    _turn_telemetry("[BARGE-IN] Cancelled active utterance task.", call_sid=call_sid)
                                is_bot_turn = False
                                speech_active = True
                                last_speech_time = now
                                audio_frames.clear()
                                audio_frames.extend(audio)
                                # Drain any stale frames from queue
                                while not audio_queue.empty():
                                    try:
                                        audio_queue.get_nowait()
                                        audio_queue.task_done()
                                    except (asyncio.QueueEmpty, ValueError):
                                        break
                        else:
                            if energy >= settings.vad_rms_threshold:
                                speech_active = True
                                last_speech_time = now
                            if speech_active:
                                audio_frames.extend(audio)
                                if (
                                    last_speech_time is not None
                                    and now - last_speech_time >= settings.vad_silence_seconds
                                    and audio_frames
                                ):
                                    is_bot_turn = True
                                    barge_in_event.clear()
                                    captured_audio = bytes(audio_frames)
                                    duration_seconds = len(captured_audio) / (sample_rate * 2)
                                    _turn_telemetry(
                                        "[VAD] User stopped speaking (silence detected). "
                                        f"Audio duration: {duration_seconds:.2f}s",
                                        call_sid=call_sid,
                                    )
                                    utterance_task = asyncio.create_task(
                                        process_utterance(captured_audio),
                                        name="sarvam-exotel-loopback",
                                    )
                                    logger.info(
                                        "loopback_utterance_detected",
                                        extra={"call_sid": call_sid, "event": "loopback"},
                                    )
                    except Exception as exc:
                        _stage_error("VAD", exc, call_sid=call_sid)
                        audio_frames.clear()
                        speech_active = False
                        last_speech_time = None
                await process_audio_frame(
                    audio,
                    AudioFrameInfo(
                        encoding="pcm_s16le",
                        sample_rate_hz=sample_rate,
                        byte_count=len(audio),
                    ),
                    call_sid,
                )
            except Exception as exc:
                # A bad frame or STT adapter must not abandon queued audio.
                _stage_error("AUDIO_WORKER", exc, call_sid=call_sid)
            finally:
                audio_queue.task_done()

    consumer_task = asyncio.create_task(consume_audio(), name="exotel-audio-consumer")
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                disconnect_code = message.get("code")
                disconnect_reason = message.get("reason")
                logger.warning(
                    "exotel_transport_disconnected",
                    extra={
                        "call_sid": call_sid,
                        "event": "disconnect",
                        "stream_sid": stream_sid,
                        "status": disconnect_code,
                    },
                )
                _turn_telemetry(
                    "[EXOTEL DISCONNECT] "
                    f"code={disconnect_code!r}, reason={disconnect_reason!r}",
                    call_sid=call_sid,
                )
                break
            if message.get("bytes") is not None:
                logger.warning(
                    "unexpected_binary_stream_frame",
                    extra={"call_sid": call_sid, "event": "media"},
                )
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                packet = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("invalid_stream_json", extra={"call_sid": call_sid, "event": "stream_error"})
                continue
            if not isinstance(packet, dict):
                logger.warning("invalid_stream_packet", extra={"call_sid": call_sid, "event": "stream_error"})
                continue
            event = packet.get("event")
            if event != "media":
                print(f"[EXOTEL RAW PACKET] {text}", flush=True)
            if event == "connected":
                logger.info("exotel_stream_connected", extra={"call_sid": call_sid, "event": "connected"})
            elif event == "start":
                start = packet.get("start")
                metadata = start if isinstance(start, dict) else packet
                call_sid = (
                    metadata.get("call_sid")
                    or metadata.get("callSid")
                    or metadata.get("CallSid")
                    or packet.get("call_sid")
                    or packet.get("callSid")
                    or call_sid
                    or "unknown"
                )
                stream_sid = (
                    metadata.get("stream_sid")
                    or metadata.get("streamSid")
                    or packet.get("stream_sid")
                    or packet.get("streamSid")
                    or stream_sid
                )
                # Exotel normally supplies snake_case fields inside `start`.
                raw_start = packet.get("start", {})
                media_format = (
                    raw_start.get("media_format", {})
                    if isinstance(raw_start, dict)
                    else {}
                )
                if not isinstance(media_format, dict):
                    media_format = {}
                # Accept the observed camelCase/root variants as a fallback.
                media_format = (
                    media_format
                    or metadata.get("mediaFormat")
                    or metadata.get("media_format")
                    or {}
                )
                if not isinstance(media_format, dict):
                    media_format = {}
                from_number = (
                    metadata.get("from")
                    or metadata.get("From")
                    or packet.get("from")
                    or packet.get("From")
                    or ""
                )
                if call_sid:
                    session = _get_session(call_sid)
                    if from_number:
                        session["caller_number"] = from_number

                sample_rate = _sample_rate_from_media_format(media_format)
                sample_width = _sample_width_from_media_format(media_format)
                encoding = str(media_format.get("encoding", "audio/x-raw")).lower()
                if encoding == "audio/x-mulaw":
                    # Decoding produces 16-bit PCM regardless of source metadata.
                    sample_width = 2
                if stream_sid:
                    outbound_sender = _ExotelOutboundSender(websocket, stream_sid)
                else:
                    logger.error(
                        "exotel_start_missing_stream_sid",
                        extra={"call_sid": call_sid, "event": "start", "start": metadata},
                    )
                logger.info(
                    "exotel_stream_started",
                    extra={
                        "call_sid": call_sid,
                        "event": "start",
                        "start": metadata,
                        "media_format": media_format,
                    },
                )
                if outbound_sender and stream_sid:
                    try:
                        await start_idle_keepalive()
                        # Immediately greet the caller with Turn-0 intake question in default language
                        greeting_text = _intake_question(0, active_caller_language)
                        if greeting_text:
                            try:
                                greet_wav = await synthesize_speech_sarvam(
                                    greeting_text,
                                    target_language_code=active_caller_language,
                                    speech_sample_rate=sample_rate,
                                )
                                greet_pcm = sarvam_audio_to_pcm16(greet_wav, sample_rate)
                                if greet_pcm:
                                    await _stream_pcm_to_exotel(
                                        websocket, stream_sid, greet_pcm, sample_rate, outbound_sender
                                    )
                                    print(f"[KAUSHAL VAANI GREETING] Played Turn-0 question in {active_caller_language}", flush=True)
                            except Exception as greet_exc:
                                _stage_error("GREETING", greet_exc, call_sid=call_sid)
                    except Exception as exc:
                        # The receive loop remains active even if an outbound
                        # keepalive frame fails during startup.
                        _stage_error("STARTUP", exc, call_sid=call_sid)
            elif event == "media":
                media = packet.get("media")
                if not isinstance(media, dict) or not isinstance(media.get("payload"), str):
                    logger.warning("invalid_media_packet", extra={"call_sid": call_sid, "event": "media"})
                    continue
                frames_received += 1
                media_chunk_count += 1
                media_item = {
                    "payload": media["payload"],
                    "sequence": media.get(
                        "sequence_number",
                        media.get(
                            "chunk",
                            packet.get("sequence_number", packet.get("chunk")),
                        ),
                    ),
                }
                try:
                    audio_queue.put_nowait(media_item)
                except asyncio.QueueFull:
                    # Drop oldest queued chunk to maintain zero-latency real-time stream
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except (asyncio.QueueEmpty, ValueError):
                        pass
                    try:
                        audio_queue.put_nowait(media_item)
                    except asyncio.QueueFull:
                        pass
                if media_chunk_count % 50 == 0:
                    logger.info(
                        "exotel_media_queued",
                        extra={"call_sid": call_sid, "event": "media", "chunk_count": media_chunk_count},
                    )
            elif event == "stop":
                stop_details = packet.get("stop")
                logger.warning(
                    "exotel_stream_stopped_by_remote",
                    extra={
                        "call_sid": call_sid,
                        "event": "stop",
                        "stream_sid": stream_sid,
                        "stop": stop_details,
                    },
                )
                _turn_telemetry(
                    f"[EXOTEL STOP] Remote stop received: {stop_details!r}",
                    call_sid=call_sid,
                )
                break
    except WebSocketDisconnect:
        logger.info("exotel_stream_disconnected", extra={"call_sid": call_sid, "event": "disconnect"})
    except Exception as exc:
        _stage_error("WEBSOCKET", exc, call_sid=call_sid)
    finally:
        try:
            await stop_idle_keepalive()
            if utterance_task and not utterance_task.done():
                utterance_task.cancel()
                with suppress(asyncio.CancelledError):
                    await utterance_task
            await audio_queue.put(None)
            await audio_queue.join()
            await consumer_task
            logger.info(
                "exotel_stream_finished_without_recording",
                extra={
                    "call_sid": call_sid,
                    "event": "stream_complete",
                    "chunk_count": media_chunk_count,
                    "stream_sid": stream_sid,
                    "frames_received": frames_received,
                    "frames_processed": frames_processed,
                    "sequence_gaps": sequence_gaps,
                },
            )
        except Exception:
            logger.exception("exotel_stream_cleanup_failed", extra={"call_sid": call_sid, "event": "stream_complete"})
        if not consumer_task.done():
            consumer_task.cancel()
            with suppress(asyncio.CancelledError):
                await consumer_task
        with suppress(Exception):
            await websocket.close()
