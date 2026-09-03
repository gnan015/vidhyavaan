"""Sarvam speech, translation, and TTS services for call recordings."""

import asyncio
import json
import logging
import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
import wave

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
_sarvam_client: httpx.AsyncClient | None = None
_sarvam_client_lock = asyncio.Lock()
_sarvam_sync_client: httpx.Client | None = None
_sarvam_sync_client_lock = Lock()


async def get_sarvam_client() -> httpx.AsyncClient:
    """Return a process-wide keep-alive client for low-latency Sarvam calls."""
    global _sarvam_client
    async with _sarvam_client_lock:
        if _sarvam_client is None or _sarvam_client.is_closed:
            settings = get_settings()
            _sarvam_client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.sarvam_request_timeout_seconds),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return _sarvam_client


async def close_sarvam_client() -> None:
    global _sarvam_client, _sarvam_sync_client
    if _sarvam_client is not None:
        await _sarvam_client.aclose()
        _sarvam_client = None
    if _sarvam_sync_client is not None:
        client = _sarvam_sync_client
        _sarvam_sync_client = None
        await asyncio.to_thread(client.close)


def _get_sarvam_sync_client() -> httpx.Client:
    """Thread-safe HTTP client for latency-sensitive live STT requests."""
    global _sarvam_sync_client
    with _sarvam_sync_client_lock:
        if _sarvam_sync_client is None or _sarvam_sync_client.is_closed:
            settings = get_settings()
            _sarvam_sync_client = httpx.Client(
                timeout=httpx.Timeout(settings.sarvam_request_timeout_seconds),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return _sarvam_sync_client


def _output_path(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}_transcript.json")


def _base_result(audio_path: Path) -> dict[str, Any]:
    return {
        "call_sid": audio_path.stem.removeprefix("call_"),
        "audio_file": audio_path.name,
        "detected_language": None,
        "original_transcript": None,
        "english_script": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }


def _write_result(output_path: Path, result: dict[str, Any]) -> None:
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def _request_sarvam(
    client: httpx.AsyncClient, audio_path: Path, api_key: str, model: str, mode: str
) -> dict[str, Any]:
    """Submit a WAV as multipart form data for one Saaras output mode."""
    with audio_path.open("rb") as audio_file:
        response = await client.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": api_key},
            files={"file": (audio_path.name, audio_file, "audio/wav")},
            data={"model": model, "mode": mode, "language_code": "unknown"},
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("transcript"), str):
        raise ValueError("Sarvam response did not include a text transcript")
    return payload


def _translate_wav_to_english_sync(wav_audio: bytes) -> dict[str, Any]:
    """Perform live STT in a worker thread so WebSocket keepalives can run."""
    if not wav_audio:
        return {"english_query": "", "detected_language_code": "en-IN"}
    settings = get_settings()
    if not settings.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is not configured")
    client = _get_sarvam_sync_client()
    response = client.post(
        SARVAM_STT_URL,
        headers={"api-subscription-key": settings.sarvam_api_key},
        files={"file": ("utterance.wav", wav_audio, "audio/wav")},
        data={
            "model": settings.sarvam_stt_model,
            "mode": "translate",
            "language_code": "unknown",
        },
    )
    response.raise_for_status()
    payload = response.json()
    transcript = payload.get("transcript") if isinstance(payload, dict) else None
    if not isinstance(transcript, str):
        raise ValueError("Sarvam translation response did not include a transcript")
    return {
        "english_query": transcript.strip(),
        "detected_language_code": payload.get("language_code", "en-IN"),
    }


async def translate_wav_to_english(wav_audio: bytes) -> dict[str, Any]:
    """Translate an in-memory utterance without blocking the ASGI event loop."""
    return await asyncio.to_thread(_translate_wav_to_english_sync, wav_audio)


async def translate_wav_bytes(wav_audio: bytes) -> str:
    """Backward-compatible English-only wrapper for recording consumers."""
    return (await translate_wav_to_english(wav_audio))["english_query"]


async def translate_english_text(text: str, target_language_code: str) -> str:
    """Translate an English answer into the caller's native language."""
    if not text.strip() or target_language_code == "en-IN":
        return text
    settings = get_settings()
    if not settings.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is not configured")
    client = await get_sarvam_client()
    response = await client.post(
        SARVAM_TRANSLATE_URL,
        headers={"api-subscription-key": settings.sarvam_api_key},
        json={
            "input": text[:2000],
            "source_language_code": "en-IN",
            "target_language_code": target_language_code,
            "model": "sarvam-translate:v1",
        },
    )
    response.raise_for_status()
    payload = response.json()
    translated = payload.get("translated_text") if isinstance(payload, dict) else None
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("Sarvam text translation response did not include text")
    return translated.strip()


async def synthesize_english_speech(
    text: str, sample_rate: int, language_code: str = "en-IN"
) -> bytes:
    """Generate a WAV response at the requested telephony sample rate."""
    if not text.strip():
        return b""
    if sample_rate not in {8000, 16000}:
        raise ValueError("Only 8 kHz and 16 kHz TTS playback is supported")
    settings = get_settings()
    if not settings.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is not configured")
    client = await get_sarvam_client()
    response = await client.post(
        SARVAM_TTS_URL,
        headers={
            "api-subscription-key": settings.sarvam_api_key,
            "Content-Type": "application/json",
        },
        json={
            "inputs": [text],
            "target_language_code": language_code,
            "speaker": settings.sarvam_tts_speaker,
            "model": settings.sarvam_tts_model,
            "speech_sample_rate": sample_rate,
            "enable_preprocessing": True,
            "pace": 1.0,
        },
    )
    if response.is_error:
        logger.error(
            "sarvam_tts_request_rejected status=%s detail=%s",
            response.status_code,
            response.text[:1000],
        )
    response.raise_for_status()
    payload = response.json()
    audios = payload.get("audios") if isinstance(payload, dict) else None
    if not isinstance(audios, list) or not audios or not isinstance(audios[0], str):
        raise ValueError("Sarvam TTS response did not include audio")
    try:
        return base64.b64decode(audios[0], validate=True)
    except ValueError as exc:
        raise ValueError("Sarvam TTS audio was not valid base64") from exc


def tts_wav_to_pcm16(wav_audio: bytes, target_sample_rate: int) -> bytes:
    """Extract mono PCM16 from Sarvam's WAV response for Exotel media frames."""
    try:
        with wave.open(io.BytesIO(wav_audio), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            source_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError("Sarvam TTS response was not a valid WAV file") from exc
    if channels != 1 or sample_width != 2:
        raise ValueError("Sarvam TTS must return mono 16-bit PCM WAV audio")
    if source_rate != target_sample_rate:
        raise ValueError(
            f"Sarvam TTS returned {source_rate} Hz, expected {target_sample_rate} Hz"
        )
    return frames


async def transcribe_and_translate_audio(audio_path: str) -> dict[str, Any]:
    """Create a transcript JSON next to ``audio_path`` using Sarvam Saaras v3.

    Two requests deliberately preserve both requested views: ``transcribe`` gives
    native-script speech and ``translate`` gives English for downstream RAG/LLM.
    Errors are persisted as structured output and returned instead of escaping the
    WebSocket teardown task.
    """
    source = Path(audio_path)
    result = _base_result(source)
    output = _output_path(source)
    try:
        if not source.is_file():
            raise FileNotFoundError("Recording file does not exist")
        if source.stat().st_size == 0:
            raise ValueError("Recording file is empty")

        settings = get_settings()
        if not settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured")

        client = await get_sarvam_client()
        transcription = await _request_sarvam(
            client, source, settings.sarvam_api_key, settings.sarvam_stt_model, "transcribe"
        )
        translation = await _request_sarvam(
            client, source, settings.sarvam_api_key, settings.sarvam_stt_model, "translate"
        )

        result.update(
            {
                "status": "completed",
                "detected_language": transcription.get("language_code"),
                "language_probability": transcription.get("language_probability"),
                "original_transcript": transcription["transcript"],
                "english_script": translation["transcript"],
                "sarvam_request_ids": {
                    "transcribe": transcription.get("request_id"),
                    "translate": translation.get("request_id"),
                },
            }
        )
        logger.info(
            "sarvam_transcription_completed",
            extra={"recording_path": str(source), "event": "transcription"},
        )
    except httpx.TimeoutException:
        result["error"] = "Sarvam request timed out"
    except httpx.HTTPStatusError as exc:
        # Includes 429 rate limits and non-retryable invalid-audio responses.
        result["error"] = f"Sarvam HTTP {exc.response.status_code}"
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
        result["error"] = str(exc)
    except Exception:
        logger.exception("sarvam_transcription_unexpected_error")
        result["error"] = "Unexpected transcription error"

    try:
        _write_result(output, result)
    except OSError:
        logger.exception("sarvam_transcript_write_failed", extra={"recording_path": str(output)})
    if result["status"] != "completed":
        logger.warning(
            "sarvam_transcription_failed",
            extra={"recording_path": str(source), "event": "transcription"},
        )
    return result
