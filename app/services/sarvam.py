"""Sarvam AI speech services (STT & TTS) for the Exotel voice pipeline."""

import asyncio
import base64
import io
import json
import logging
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

_sarvam_client: httpx.AsyncClient | None = None
_sarvam_client_lock = asyncio.Lock()

# Valid speakers for Sarvam bulbul:v3
_BULBUL_V3_SPEAKERS = {
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand",
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali"
}

_LANGUAGE_DEFAULT_SPEAKERS = {
    "te-IN": "kavitha",
    "hi-IN": "priya",
    "ta-IN": "gokul",
    "kn-IN": "kavitha",
    "bn-IN": "roopa",
    "mr-IN": "rupali",
    "en-IN": "shreya",
}


async def get_sarvam_client() -> httpx.AsyncClient:
    """Return a shared keep-alive httpx client for Sarvam AI requests."""
    global _sarvam_client
    loop = asyncio.get_running_loop()
    async with _sarvam_client_lock:
        if (
            _sarvam_client is None
            or _sarvam_client.is_closed
            or getattr(_sarvam_client, "_bound_loop", None) is not loop
        ):
            if _sarvam_client is not None and not _sarvam_client.is_closed:
                await _sarvam_client.aclose()
            _sarvam_client = httpx.AsyncClient(
                timeout=httpx.Timeout(45.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
            setattr(_sarvam_client, "_bound_loop", loop)
        return _sarvam_client


async def close_sarvam_client() -> None:
    """Close the shared Sarvam HTTP client upon server shutdown."""
    global _sarvam_client
    async with _sarvam_client_lock:
        client, _sarvam_client = _sarvam_client, None
    if client is not None and not client.is_closed:
        await client.aclose()


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 8000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit linear PCM mono frames in a standard RIFF/WAVE container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _resolve_speaker(speaker: str | None, target_language_code: str) -> str:
    """Resolve requested speaker or alias (e.g. legacy 'meera') to a valid bulbul:v3 voice."""
    settings = get_settings()
    lang = target_language_code or settings.default_caller_language or "te-IN"

    if speaker and speaker.lower() in _BULBUL_V3_SPEAKERS:
        return speaker.lower()

    if speaker and speaker.lower() == "meera":
        # 'meera' was an early Sarvam voice name, map to modern primary Indic voice
        return _LANGUAGE_DEFAULT_SPEAKERS.get(lang, "kavitha")

    configured = getattr(settings, "sarvam_speaker", "kavitha")
    if configured and configured.lower() in _BULBUL_V3_SPEAKERS:
        return configured.lower()

    return _LANGUAGE_DEFAULT_SPEAKERS.get(lang, "kavitha")


def _resolve_tts_model(model: str | None) -> str:
    """Normalize TTS model name to active Sarvam version."""
    settings = get_settings()
    chosen = model or getattr(settings, "sarvam_tts_model", "bulbul:v3")
    if chosen in ("bulbul:v1", "bulbul:v2"):
        # Deprecated by Sarvam in favor of bulbul:v3
        return "bulbul:v3"
    return chosen


def _resolve_stt_model(model: str | None) -> str:
    """Normalize STT model name to active Sarvam version."""
    settings = get_settings()
    chosen = model or getattr(settings, "sarvam_stt_model", "saaras:v3")
    if chosen in ("saaras:v1", "saaras:v2"):
        return "saaras:v3"
    return chosen


async def transcribe_audio_sarvam(
    wav_bytes: bytes,
    language_code: str = "te-IN",
    model: str | None = None,
) -> dict[str, Any]:
    """Transcribe audio using Sarvam AI's Speech-to-Text API.

    Args:
        wav_bytes: Audio bytes. Either a full WAV container or raw PCM16 mono.
        language_code: BCP-47 language tag (e.g. 'te-IN') or 'unknown' for auto-detection.
        model: Optional model override (defaults to saaras:v3).

    Returns:
        A dict containing:
          - transcript: The transcribed text string.
          - language_code: The detected/confirmed language code.
          - english_query: Compatible alias for transcript.
          - detected_language_code: Compatible alias for language_code.
    """
    settings = get_settings()
    api_key = settings.sarvam_api_key
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not configured in settings or .env")

    # If raw PCM16 was passed, convert it to a valid WAV container
    if not (len(wav_bytes) >= 12 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"):
        wav_payload = pcm16_to_wav(wav_bytes, sample_rate=8000, channels=1)
    else:
        wav_payload = wav_bytes

    chosen_model = _resolve_stt_model(model)
    target_lang = language_code or "unknown"

    client = await get_sarvam_client()
    headers = {"api-subscription-key": api_key}
    files = {"file": ("audio.wav", wav_payload, "audio/wav")}
    data = {"model": chosen_model, "language_code": target_lang}

    try:
        response = await client.post(SARVAM_STT_URL, headers=headers, files=files, data=data)
        if response.status_code == 400 and chosen_model != "saaras:v3":
            # Retry with saaras:v3 if specific model was rejected
            data["model"] = "saaras:v3"
            response = await client.post(SARVAM_STT_URL, headers=headers, files=files, data=data)

        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("sarvam_stt_http_error status=%s body=%s", exc.response.status_code, exc.response.text)
        raise
    except Exception as exc:
        logger.error("sarvam_stt_failed error=%s", exc)
        raise

    payload = response.json()
    transcript = str(payload.get("transcript") or "").strip()
    detected_lang = str(payload.get("language_code") or target_lang or "te-IN")

    return {
        "transcript": transcript,
        "language_code": detected_lang,
        "english_query": transcript,
        "detected_language_code": detected_lang,
    }


async def synthesize_speech_sarvam(
    text: str,
    target_language_code: str = "te-IN",
    speaker: str = "meera",
    speech_sample_rate: int = 8000,
    model: str | None = None,
) -> bytes:
    """Synthesize speech using Sarvam AI's Text-to-Speech API.

    Args:
        text: Text to synthesize.
        target_language_code: Target BCP-47 language tag (e.g. 'te-IN').
        speaker: Voice name (e.g. 'kavitha', 'priya', or legacy 'meera').
        speech_sample_rate: Telephony sample rate (default 8000 Hz).
        model: TTS model name (defaults to bulbul:v3).

    Returns:
        Raw .wav bytes decoded from the response.
    """
    settings = get_settings()
    api_key = settings.sarvam_api_key
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not configured in settings or .env")

    cleaned_text = text.strip()
    if not cleaned_text:
        return b""

    chosen_speaker = _resolve_speaker(speaker, target_language_code)
    chosen_model = _resolve_tts_model(model)

    payload = {
        "inputs": [cleaned_text],
        "target_language_code": target_language_code,
        "speaker": chosen_speaker,
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.0,
        "speech_sample_rate": speech_sample_rate,
        "enable_preprocessing": True,
        "model": chosen_model,
    }

    client = await get_sarvam_client()
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }

    try:
        response = await client.post(SARVAM_TTS_URL, headers=headers, json=payload)
        if response.status_code == 400:
            err_msg = response.text
            logger.warning("sarvam_tts_rejected detail=%s, retrying with standard bulbul:v3 defaults", err_msg)
            # Retry with fallback speaker and model
            payload["model"] = "bulbul:v3"
            payload["speaker"] = _LANGUAGE_DEFAULT_SPEAKERS.get(target_language_code, "kavitha")
            response = await client.post(SARVAM_TTS_URL, headers=headers, json=payload)

        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("sarvam_tts_http_error status=%s body=%s", exc.response.status_code, exc.response.text)
        raise
    except Exception as exc:
        logger.error("sarvam_tts_failed error=%s", exc)
        raise

    res_json = response.json()
    audios = res_json.get("audios")
    if not audios or not isinstance(audios, list):
        raise ValueError("Sarvam TTS response did not contain 'audios' list")

    wav_bytes = base64.b64decode(audios[0])
    return wav_bytes


def _resample_audio_polyphase(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample float32 audio using scipy or linear interpolation fallback."""
    if source_rate == target_rate or len(samples) == 0:
        return samples
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(source_rate, target_rate)
        up = target_rate // g
        down = source_rate // g
        return resample_poly(samples, up, down).astype(np.float32)
    except Exception:
        num_target = int(len(samples) * target_rate / source_rate)
        orig_idx = np.linspace(0, len(samples) - 1, len(samples))
        target_idx = np.linspace(0, len(samples) - 1, num_target)
        return np.interp(target_idx, orig_idx, samples).astype(np.float32)


def sarvam_audio_to_pcm16(wav_bytes: bytes, target_sample_rate: int = 8000) -> bytes:
    """Unpack WAV bytes and convert them into 16-bit linear PCM mono frames matching Exotel's sample rate."""
    if not wav_bytes:
        return b""

    # Parse WAV container
    if len(wav_bytes) >= 44 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
        fmt_pos = wav_bytes.find(b"fmt ")
        data_pos = wav_bytes.find(b"data")
        if fmt_pos != -1 and data_pos != -1:
            try:
                fmt_len = struct.unpack("<I", wav_bytes[fmt_pos + 4 : fmt_pos + 8])[0]
                fmt_data = wav_bytes[fmt_pos + 8 : fmt_pos + 8 + fmt_len]
                fmt_tag, channels, source_rate, _, _, bits_per_sample = struct.unpack(
                    "<HHIIHH", fmt_data[:16]
                )
                data_len = struct.unpack("<I", wav_bytes[data_pos + 4 : data_pos + 8])[0]
                raw_data = wav_bytes[data_pos + 8 : data_pos + 8 + data_len]

                floats: np.ndarray | None = None
                if fmt_tag == 3 and bits_per_sample == 32:
                    floats = np.frombuffer(raw_data, dtype=np.float32).copy()
                    if channels == 2:
                        floats = (floats[0::2] + floats[1::2]) * 0.5
                elif fmt_tag == 1 and bits_per_sample == 16:
                    ints = np.frombuffer(raw_data, dtype=np.int16)
                    if channels == 2:
                        ints_mono = (ints[0::2].astype(np.float32) + ints[1::2].astype(np.float32)) * 0.5
                        floats = ints_mono / 32768.0
                    else:
                        floats = ints.astype(np.float32) / 32768.0

                if floats is not None and len(floats) > 0:
                    # Remove DC bias to prevent pops, clicks, and asymmetric distortion on phone lines
                    floats = floats - np.mean(floats)
                    # Resample if needed
                    if source_rate != target_sample_rate and target_sample_rate > 0:
                        floats = _resample_audio_polyphase(floats, source_rate, target_sample_rate)
                    # Safe peak headroom normalization without over-amplification
                    peak = float(np.max(np.abs(floats))) if len(floats) else 0.0
                    if peak > 0.95:
                        floats = floats * (0.90 / peak)
                    pcm16 = np.clip(floats * 32767.0, -32768, 32767).astype(np.int16)
                    return pcm16.tobytes()
            except Exception as exc:
                logger.warning("sarvam_numpy_wav_resample_failed error=%s", exc)

    # Standard library wave reader fallback
    try:
        if len(wav_bytes) >= 12 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
            with wave.open(io.BytesIO(wav_bytes), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                source_rate = source.getframerate()
                frames = source.readframes(source.getnframes())
                if sample_width == 2:
                    ints = np.frombuffer(frames, dtype=np.int16)
                    if channels == 2:
                        floats = (ints[0::2].astype(np.float32) + ints[1::2].astype(np.float32)) * 0.5 / 32768.0
                    else:
                        floats = ints.astype(np.float32) / 32768.0
                    floats = floats - np.mean(floats)
                    if source_rate != target_sample_rate and target_sample_rate > 0:
                        floats = _resample_audio_polyphase(floats, source_rate, target_sample_rate)
                    peak = float(np.max(np.abs(floats))) if len(floats) else 0.0
                    if peak > 0.95:
                        floats = floats * (0.90 / peak)
                    return np.clip(floats * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    except Exception as exc:
        logger.warning("sarvam_wave_reader_failed error=%s", exc)

    usable_len = len(wav_bytes) - (len(wav_bytes) % 2)
    return wav_bytes[:usable_len]


async def transcribe_audio_file_sarvam(audio_path: str) -> dict[str, Any]:
    """Persist Sarvam STT transcription for a saved call recording WAV file."""
    source = Path(audio_path)
    result: dict[str, Any] = {
        "call_sid": source.stem.removeprefix("call_"),
        "audio_file": source.name,
        "detected_language": None,
        "original_transcript": None,
        "english_script": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }
    try:
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("Recording file does not exist or is empty")
        wav_data = source.read_bytes()
        recognition = await transcribe_audio_sarvam(wav_data)
        result.update(
            {
                "status": "completed",
                "detected_language": recognition["language_code"],
                "original_transcript": recognition["transcript"],
                "english_script": recognition["transcript"],
            }
        )
        logger.info("sarvam_transcription_completed", extra={"event": "transcription", "recording_path": str(source)})
    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("sarvam_transcription_failed", extra={"event": "transcription", "recording_path": str(source)})

    output = source.with_name(f"{source.stem}_transcript.json")
    try:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.exception("sarvam_transcript_write_failed", extra={"recording_path": str(output)})
    return result
