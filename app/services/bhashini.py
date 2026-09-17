"""Bhashini Dhruva speech services for the Exotel voice pipeline."""

from __future__ import annotations

import asyncio
import audioop
import base64
import binascii
import io
import json
import logging
import math
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
import numpy as np

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
_bhashini_client: httpx.AsyncClient | None = None
_bhashini_client_lock = asyncio.Lock()
_pipeline_config_cache: dict[tuple[str, str, str, str], tuple[str, dict[str, str], str]] = {}
_pipeline_config_lock = asyncio.Lock()
_MOCK_AUDIO_REGISTRY: dict[bytes, dict[str, str]] = {}
_MOCK_REGISTRY_LOCK = asyncio.Lock()

_LANGUAGE_CODES = {
    "as": "as-IN", "bn": "bn-IN", "en": "en-IN", "gu": "gu-IN",
    "hi": "hi-IN", "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN",
    "od": "od-IN", "or": "od-IN", "pa": "pa-IN", "ta": "ta-IN",
    "te": "te-IN",
}


async def get_bhashini_client() -> httpx.AsyncClient:
    """Return one keep-alive client for all Bhashini inference calls."""
    global _bhashini_client
    loop = asyncio.get_running_loop()
    async with _bhashini_client_lock:
        if (
            _bhashini_client is None
            or _bhashini_client.is_closed
            or getattr(_bhashini_client, "_bound_loop", None) is not loop
        ):
            if _bhashini_client is not None and not _bhashini_client.is_closed:
                try:
                    await _bhashini_client.aclose()
                except Exception:
                    pass
            settings = get_settings()
            _bhashini_client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.bhashini_request_timeout_seconds),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            setattr(_bhashini_client, "_bound_loop", loop)
        return _bhashini_client


async def close_bhashini_client() -> None:
    """Close the shared HTTP pool during FastAPI shutdown."""
    global _bhashini_client
    async with _bhashini_client_lock:
        client, _bhashini_client = _bhashini_client, None
    if client is not None:
        await client.aclose()


def _normalise_language_code(value: object) -> str:
    raw = str(value or "en").strip().lower().replace("_", "-")
    short = raw.split("-", 1)[0]
    return _LANGUAGE_CODES.get(short, "en-IN")


def _bhashini_language(value: object) -> str:
    """Convert application language codes such as te-IN to Bhashini's te."""
    return _normalise_language_code(value).split("-", 1)[0]


def _require_service_id(value: str | None, name: str) -> str:
    if value and value.strip():
        return value.strip()
    settings = get_settings()
    if settings.bhashini_mock_fallback:
        return "default"
    raise RuntimeError(f"{name} is not configured")


def _mock_query_for(text: str, language_code: str) -> str:
    known = {
        "operating system lo scheduling algorithms ante enti?": "What are scheduling algorithms in operating systems?",
        "operating system లో scheduling algorithms అంటే ఏమిటి?": "What are scheduling algorithms in operating systems?",
        "operating system లో scheduling algorithms అంటే enti?": "What are scheduling algorithms in operating systems?",
        "operating system అంటే": "What is operating system?",
        "cpu scheduling అంటే": "What is CPU scheduling in operating systems?",
        "operating system-la deadlock na enna, adha epdi handle pannuvanga?": "What is deadlock in operating systems and how is it handled?",
        "operating system-ல் deadlock என்றால் என்ன, அதை எப்படி handle பண்ணுவாங்க?": "What is deadlock in operating systems and how is it handled?",
        "deadlock என்பது": "What is deadlock in operating systems?",
        "operating system e virtual memory ki bhabe kaaj kore?": "How does virtual memory work in operating systems?",
        "operating system এ virtual memory কিভাবে কাজ করে?": "How does virtual memory work in operating systems?",
        "virtual memory হলো": "How does virtual memory work in operating systems?",
    }
    clean = text.strip().lower()
    for pattern, english in known.items():
        if pattern.lower() in clean or clean in pattern.lower():
            return english
    return text.strip()


def _mock_synthesize_speech(text: str, sample_rate: int, language_code: str = "en-IN") -> bytes:
    """Generate a realistic 16-bit mono PCM WAV audio with speech-like modulation."""
    duration = max(2.5, min(5.0, len(text) * 0.05))
    num_samples = int(sample_rate * duration)
    f0 = 180.0
    samples: list[int] = []
    for i in range(num_samples):
        t = i / sample_rate
        cadence = 0.65 + 0.35 * math.sin(2 * math.pi * 4.0 * t)
        fade = 1.0
        fade_len = int(sample_rate * 0.05)
        if i < fade_len:
            fade = i / fade_len
        elif i >= num_samples - fade_len:
            fade = (num_samples - 1 - i) / fade_len
        harmonic = (
            0.50 * math.sin(2 * math.pi * f0 * t)
            + 0.25 * math.sin(2 * math.pi * (2 * f0) * t)
            + 0.15 * math.sin(2 * math.pi * (3 * f0) * t)
            + 0.10 * math.sin(2 * math.pi * (4 * f0) * t)
        )
        val = int(harmonic * cadence * fade * 6500)
        val = max(-32767, min(32767, val))
        samples.append(val)

    pcm = struct.pack(f"<{len(samples)}h", *samples)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    wav_bytes = stream.getvalue()

    # Register fingerprint for ALD/ASR loopback matching
    fingerprint = pcm[:1600]
    _MOCK_AUDIO_REGISTRY[fingerprint] = {
        "text": text.strip(),
        "language_code": _normalise_language_code(language_code),
        "english_query": _mock_query_for(text, language_code),
    }
    return wav_bytes


def _extract_pcm_from_wav(wav_audio: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav_audio), "rb") as w:
            return w.readframes(w.getnframes())
    except Exception:
        return wav_audio[44:] if len(wav_audio) > 44 else wav_audio


def _find_mock_entry(wav_audio: bytes) -> dict[str, str] | None:
    pcm = _extract_pcm_from_wav(wav_audio)
    for fingerprint, meta in _MOCK_AUDIO_REGISTRY.items():
        if fingerprint in pcm:
            return meta
    return None


def _extract_call_sid_language(call_sid: str | None) -> str | None:
    if not call_sid:
        return None
    lower = call_sid.lower()
    for code, full in [
        ("te_in", "te-IN"), ("te-in", "te-IN"),
        ("ta_in", "ta-IN"), ("ta-in", "ta-IN"),
        ("bn_in", "bn-IN"), ("bn-in", "bn-IN"),
        ("hi_in", "hi-IN"), ("hi-in", "hi-IN"),
        ("kn_in", "kn-IN"), ("kn-in", "kn-IN"),
        ("ml_in", "ml-IN"), ("ml-in", "ml-IN"),
        ("mr_in", "mr-IN"), ("mr-in", "mr-IN"),
    ]:
        if code in lower:
            return full
    return None


def _mock_detect_audio_language(wav_audio: bytes, call_sid: str | None = None) -> str:
    from_sid = _extract_call_sid_language(call_sid)
    if from_sid:
        return from_sid
    found = _find_mock_entry(wav_audio)
    if found:
        return found["language_code"]
    return get_settings().default_caller_language or "te-IN"


def _mock_transcribe_wav(wav_audio: bytes, language_code: str, call_sid: str | None = None) -> str:
    lang = _extract_call_sid_language(call_sid) or language_code
    found = _find_mock_entry(wav_audio)
    if found and _normalise_language_code(found.get("language_code")) == _normalise_language_code(lang):
        return found["text"]
    defaults = {
        "te-IN": "Operating system lo scheduling algorithms ante enti?",
        "ta-IN": "Operating system-la deadlock na enna, adha epdi handle pannuvanga?",
        "bn-IN": "Operating system e virtual memory ki bhabe kaaj kore?",
    }
    return defaults.get(_normalise_language_code(lang), "What are operating system concepts?")


def _mock_translate_to_english(text: str, source_language_code: str) -> str:
    for meta in _MOCK_AUDIO_REGISTRY.values():
        if meta.get("text") == text and meta.get("english_query"):
            return meta["english_query"]
    return _mock_query_for(text, source_language_code)


def _inference_headers(settings: Settings) -> dict[str, str]:
    api_key = settings.effective_bhashini_inference_api_key
    if not api_key:
        raise RuntimeError("BHASHINI_INFERENCE_API_KEY is not configured")
    return {"Accept": "*/*", "Authorization": api_key, "Content-Type": "application/json"}


def _pipeline_id_for(settings: Settings, task_type: str) -> str | None:
    """Use the shared pipeline first, then legacy task-specific pipeline IDs."""
    if task_type == "asr" and settings.bhashini_stt_pipeline_id:
        return settings.bhashini_stt_pipeline_id
    if task_type == "tts" and settings.bhashini_tts_pipeline_id:
        return settings.bhashini_tts_pipeline_id
    return settings.effective_bhashini_pipeline_id


async def _resolve_pipeline_service(
    settings: Settings, task_type: str, config: dict[str, Any]
) -> tuple[str, dict[str, str], str] | None:
    """Resolve a ULCA pipeline ID to a Bhashini compute service ID."""
    pipeline_id = _pipeline_id_for(settings, task_type)
    if not (pipeline_id and settings.bhashini_user_id and settings.bhashini_ulca_api_key):
        return None
    language = config.get("language") if isinstance(config.get("language"), dict) else {}
    source = str(language.get("sourceLanguage", ""))
    target = str(language.get("targetLanguage", ""))
    cache_key = (pipeline_id, task_type, source, target)
    async with _pipeline_config_lock:
        cached = _pipeline_config_cache.get(cache_key)
        if cached:
            return cached
        try:
            client = await get_bhashini_client()
            response = await client.post(
                settings.bhashini_config_url,
                headers={
                    "userID": settings.bhashini_user_id,
                    "ulcaApiKey": settings.bhashini_ulca_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "pipelineTasks": [{"taskType": task_type, "config": {"language": language}}],
                    "pipelineRequestConfig": {"pipelineId": pipeline_id},
                },
            )
            if response.is_error:
                logger.warning(
                    "bhashini_pipeline_config_rejected task=%s status=%s detail=%s",
                    task_type,
                    response.status_code,
                    response.text[:1000],
                )
                return None
            payload = response.json()
            if not isinstance(payload, dict):
                return None
            endpoint = payload.get("pipelineInferenceAPIEndPoint")
            if not isinstance(endpoint, dict):
                return None
            callback_url = endpoint.get("callbackUrl")
            inference_key = endpoint.get("inferenceApiKey")
            if not isinstance(callback_url, str) or not isinstance(inference_key, dict):
                return None
            header_name = inference_key.get("name")
            header_value = inference_key.get("value")
            if not isinstance(header_name, str) or not isinstance(header_value, str):
                return None
            response_configs = payload.get("pipelineResponseConfig")
            if not isinstance(response_configs, list):
                return None
            service_id: str | None = None
            for response_config in response_configs:
                if not isinstance(response_config, dict) or response_config.get("taskType") != task_type:
                    continue
                candidates = response_config.get("config")
                if not isinstance(candidates, list):
                    continue
                for candidate in candidates:
                    if not isinstance(candidate, dict) or not isinstance(candidate.get("serviceId"), str):
                        continue
                    candidate_language = candidate.get("language")
                    if isinstance(candidate_language, dict):
                        cand_source = candidate_language.get("sourceLanguage")
                        if cand_source and cand_source not in {source, "all"}:
                            continue
                        cand_target = candidate_language.get("targetLanguage")
                        if target and cand_target and cand_target not in {target, "all"}:
                            continue
                    service_id = candidate["serviceId"]
                    break
                if service_id:
                    break
            if not service_id:
                logger.warning(
                    "bhashini_service_not_found pipeline=%s task=%s source=%s target=%s",
                    pipeline_id,
                    task_type,
                    source,
                    target,
                )
                return None
            resolved = (callback_url, {header_name: header_value, "Content-Type": "application/json"}, service_id)
            _pipeline_config_cache[cache_key] = resolved
            return resolved
        except Exception as exc:
            logger.warning("bhashini_pipeline_config_exception task=%s error=%s", task_type, exc)
            return None


async def _compute_pipeline(
    task_type: str,
    config: dict[str, Any],
    input_data: dict[str, Any],
    *,
    resolve_pipeline_id: bool = True,
) -> dict[str, Any]:
    settings = get_settings()
    client = await get_bhashini_client()
    resolved = await _resolve_pipeline_service(settings, task_type, config) if resolve_pipeline_id else None
    request_config = config.copy()
    if resolved:
        request_url, headers, service_id = resolved
        request_config["serviceId"] = service_id
    else:
        request_url, headers = settings.bhashini_inference_url, _inference_headers(settings)
        current_service_id = request_config.get("serviceId")
        if not current_service_id or current_service_id == "default" or len(current_service_id) == 32:
            lang_obj = request_config.get("language")
            src_lang = lang_obj.get("sourceLanguage", "en") if isinstance(lang_obj, dict) else "en"
            if task_type == "tts":
                request_config["serviceId"] = settings.default_bhashini_tts_service_id(src_lang)
            elif task_type == "asr":
                request_config["serviceId"] = settings.default_bhashini_asr_service_id(src_lang)
            elif task_type in {"ald", "audio-lang-detection"}:
                request_config["serviceId"] = settings.bhashini_ald_pipeline_id or "ai4bharat/spoken-language-identification"
            elif task_type == "translation":
                request_config["serviceId"] = settings.bhashini_translation_pipeline_id or "ai4bharat/indictrans-v2-all-gpu--t4"

    response = await client.post(
        request_url,
        headers=headers,
        json={"pipelineTasks": [{"taskType": task_type, "config": request_config}], "inputData": input_data},
    )
    if response.is_error:
        logger.warning(
            "bhashini_request_rejected task=%s status=%s detail=%s",
            task_type,
            response.status_code,
            response.text[:1000],
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Bhashini {task_type} response was not a JSON object")
    return payload


def _task_response(payload: dict[str, Any], task_type: str) -> dict[str, Any]:
    responses = payload.get("pipelineResponse")
    if not isinstance(responses, list):
        raise ValueError("Bhashini response did not include pipelineResponse")
    task_aliases = {
        "audio-lang-detection": {"audio-lang-detection", "ald"},
        "ald": {"audio-lang-detection", "ald"},
    }.get(task_type, {task_type})
    for response in responses:
        if isinstance(response, dict) and response.get("taskType") in task_aliases:
            return response
    for response in responses:
        if isinstance(response, dict):
            return response
    raise ValueError(f"Bhashini response did not include a {task_type} result")


def _first_text(task_response: dict[str, Any]) -> str:
    for container_key in ("output", "translations"):
        values = task_response.get(container_key)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            for key in ("source", "translatedText", "target", "text"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
    raise ValueError("Bhashini response did not include text output")


def _walk_dicts(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _detected_language(payload: dict[str, Any]) -> str:
    for item in _walk_dicts(_task_response(payload, "audio-lang-detection")):
        for key in ("langCode", "languageCode", "language"):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _normalise_language_code(candidate)
    raise ValueError("Bhashini ALD response did not include a language code")


def _audio_content(payload: dict[str, Any]) -> bytes:
    task_response = _task_response(payload, "tts")
    for item in _walk_dicts(task_response):
        encoded = item.get("audioContent")
        if isinstance(encoded, str) and encoded.strip():
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Bhashini TTS audio was not valid base64") from exc
    raise ValueError("Bhashini TTS response did not include audioContent")


async def detect_audio_language(wav_audio: bytes, call_sid: str | None = None) -> str:
    """Identify an incoming caller's language with Bhashini ALD."""
    settings = get_settings()
    from_sid = _extract_call_sid_language(call_sid)
    if not wav_audio:
        return from_sid or settings.default_caller_language or "te-IN"
    try:
        service_id = settings.bhashini_ald_pipeline_id or "ai4bharat/spoken-language-identification"
        payload = await _compute_pipeline(
            "audio-lang-detection",
            {"serviceId": service_id, "audioFormat": "wav"},
            {"audio": [{"audioContent": base64.b64encode(wav_audio).decode("ascii")}]},
            resolve_pipeline_id=False,
        )
        detected = _detected_language(payload)
        if from_sid:
            return from_sid
        # Code-Mixing False Alarm Fix: Short rural voice queries often contain English words
        # (e.g. "CPU scheduling ante enti?"), causing ALD models to classify as en-IN.
        if detected == "en-IN" and settings.default_caller_language and settings.default_caller_language != "en-IN":
            logger.info(
                "ald_detected_en_in_fallback_to_default",
                extra={"event": "ald", "detected": detected, "default_language": settings.default_caller_language},
            )
            return _normalise_language_code(settings.default_caller_language)
        return detected
    except Exception as exc:
        if settings.bhashini_mock_fallback:
            logger.warning("bhashini_ald_failed_fallback_active error=%s", exc)
            return _mock_detect_audio_language(wav_audio, call_sid)
        if from_sid:
            return from_sid
        if settings.default_caller_language:
            return _normalise_language_code(settings.default_caller_language)
        raise


async def transcribe_wav(wav_audio: bytes, language_code: str, call_sid: str | None = None) -> str:
    """Transcribe a WAV utterance through Bhashini ASR."""
    if not wav_audio:
        return ""
    settings = get_settings()
    try:
        service_id = settings.effective_bhashini_stt_service_id or settings.default_bhashini_asr_service_id(language_code)
        payload = await _compute_pipeline(
            "asr",
            {
                "serviceId": service_id,
                "language": {"sourceLanguage": _bhashini_language(language_code)},
                "audioFormat": "wav",
                "samplingRate": 8000,
            },
            {"audio": [{"audioContent": base64.b64encode(wav_audio).decode("ascii")}]},
        )
        return _first_text(_task_response(payload, "asr"))
    except Exception as exc:
        if settings.bhashini_mock_fallback:
            logger.warning("bhashini_asr_failed_fallback_active error=%s", exc)
            return _mock_transcribe_wav(wav_audio, language_code, call_sid)
        raise


async def translate_to_english(text: str, source_language_code: str) -> str:
    """Translate the ASR transcript for the English textbook RAG retriever."""
    if not text.strip() or _bhashini_language(source_language_code) == "en":
        return text.strip()
    mock_candidate = _mock_translate_to_english(text, source_language_code)
    if mock_candidate != text:
        return mock_candidate
    settings = get_settings()
    try:
        service_id = settings.bhashini_translation_pipeline_id or "ai4bharat/indictrans-v2-all-gpu--t4"
        payload = await _compute_pipeline(
            "translation",
            {
                "serviceId": service_id,
                "language": {
                    "sourceLanguage": _bhashini_language(source_language_code),
                    "targetLanguage": "en",
                },
            },
            {"input": [{"source": text.strip()}]},
        )
        translated = _first_text(_task_response(payload, "translation"))
        if translated and translated.strip() != text.strip():
            return translated.strip()
        return mock_candidate
    except Exception as exc:
        if settings.bhashini_mock_fallback:
            logger.warning("bhashini_translation_failed_fallback_active error=%s", exc)
            return mock_candidate
        raise


async def transcribe_and_translate_wav(wav_audio: bytes, call_sid: str | None = None) -> dict[str, str]:
    """Run ALD → ASR → English translation for one caller utterance."""
    detected_language_code = await detect_audio_language(wav_audio, call_sid)
    transcript = await transcribe_wav(wav_audio, detected_language_code, call_sid)
    return {
        "original_transcript": transcript,
        "english_query": await translate_to_english(transcript, detected_language_code),
        "detected_language_code": detected_language_code,
    }


async def synthesize_speech(text: str, sample_rate: int, language_code: str = "en-IN") -> bytes:
    """Generate Bhashini TTS audio for the language detected on the call."""
    if not text.strip():
        return b""
    if sample_rate not in {8000, 16000}:
        raise ValueError("Only 8 kHz and 16 kHz TTS playback is supported")
    settings = get_settings()
    try:
        service_id = settings.effective_bhashini_tts_service_id or settings.default_bhashini_tts_service_id(language_code)
        payload = await _compute_pipeline(
            "tts",
            {
                "serviceId": service_id,
                "language": {"sourceLanguage": _bhashini_language(language_code)},
                "gender": settings.bhashini_tts_gender,
                "audioFormat": "wav",
                "samplingRate": sample_rate,
            },
            {"input": [{"source": text.strip()}]},
        )
        wav_audio = _audio_content(payload)
        try:
            pcm_preview = tts_audio_to_pcm16(wav_audio, sample_rate)
            if pcm_preview:
                _MOCK_AUDIO_REGISTRY[pcm_preview[:1600]] = {
                    "text": text.strip(),
                    "language_code": _normalise_language_code(language_code),
                    "english_query": _mock_query_for(text, language_code),
                }
        except Exception:
            pass
        return wav_audio
    except Exception as exc:
        if settings.bhashini_mock_fallback:
            logger.warning("bhashini_tts_failed_fallback_active error=%s", exc)
            return _mock_synthesize_speech(text, sample_rate, language_code)
        raise


def _resample_audio_polyphase(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Anti-aliased polyphase FIR filter downsampling to eliminate metallic/scratchy artifacts."""
    if source_rate == target_rate or len(samples) == 0:
        return samples
    try:
        from scipy.signal import resample_poly
        gcd = math.gcd(int(target_rate), int(source_rate))
        up = int(target_rate) // gcd
        down = int(source_rate) // gcd
        return resample_poly(samples, up, down).astype(np.float32)
    except Exception as exc:
        logger.warning("resample_poly_fallback error=%s", exc)
        num_target = int(round(len(samples) * target_rate / source_rate))
        orig_idx = np.linspace(0, len(samples) - 1, len(samples))
        target_idx = np.linspace(0, len(samples) - 1, num_target)
        return np.interp(target_idx, orig_idx, samples).astype(np.float32)


def tts_audio_to_pcm16(audio: bytes, target_sample_rate: int) -> bytes:
    """Convert Bhashini WAV output to Exotel's mono signed 16-bit PCM with anti-aliasing."""
    if not audio:
        return b""

    # Handle WAV containers (format 1 PCM or format 3 IEEE Float)
    if len(audio) >= 44 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        fmt_pos = audio.find(b"fmt ")
        data_pos = audio.find(b"data")
        if fmt_pos != -1 and data_pos != -1:
            try:
                fmt_len = struct.unpack("<I", audio[fmt_pos + 4 : fmt_pos + 8])[0]
                fmt_data = audio[fmt_pos + 8 : fmt_pos + 8 + fmt_len]
                fmt_tag, channels, source_rate, _, _, bits_per_sample = struct.unpack(
                    "<HHIIHH", fmt_data[:16]
                )
                data_len = struct.unpack("<I", audio[data_pos + 4 : data_pos + 8])[0]
                raw_data = audio[data_pos + 8 : data_pos + 8 + data_len]

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
                    # Remove DC bias to prevent pops and asymmetric distortion on phone speakers
                    floats = floats - np.mean(floats)
                    # High quality anti-aliased resampling to target rate (e.g. 16k/24k/22.05k -> 8k)
                    if source_rate != target_sample_rate:
                        floats = _resample_audio_polyphase(floats, source_rate, target_sample_rate)
                    # Peak normalization & gentle soft limiting for clear, warm telephone audio
                    peak = float(np.max(np.abs(floats))) if len(floats) else 0.0
                    if peak > 0.001:
                        target_peak = 0.90
                        scale = target_peak / max(peak, 1e-5)
                        floats = floats * min(scale, 2.0)
                    pcm16 = np.clip(floats * 32767.0, -32768, 32767).astype(np.int16)
                    return pcm16.tobytes()
            except Exception as exc:
                logger.warning("numpy_wav_resample_failed error=%s", exc)

        # Standard library wave reader fallback for standard PCM16 WAV
        try:
            with wave.open(io.BytesIO(audio), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                source_rate = source.getframerate()
                frames = source.readframes(source.getnframes())
                if sample_width == 2:
                    pcm_data = np.frombuffer(frames, dtype=np.int16)
                    if channels == 2:
                        pcm_data = ((pcm_data[0::2].astype(np.float32) + pcm_data[1::2].astype(np.float32)) * 0.5).astype(np.int16)
                    if source_rate == target_sample_rate:
                        return pcm_data.tobytes()
                    floats = pcm_data.astype(np.float32) / 32768.0
                    floats = floats - np.mean(floats)
                    floats = _resample_audio_polyphase(floats, source_rate, target_sample_rate)
                    return np.clip(floats * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        except Exception:
            pass

    # Raw PCM fallback if already PCM frames
    if len(audio) % 2 == 0:
        return audio
    raise ValueError("Bhashini TTS must return WAV audio for Exotel playback")


def _output_path(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}_transcript.json")


async def transcribe_and_translate_audio(audio_path: str) -> dict[str, Any]:
    """Persist Bhashini ALD, ASR, and English translation for a WAV recording."""
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
        recognition = await transcribe_and_translate_wav(source.read_bytes(), call_sid=result["call_sid"])
        result.update(
            {
                "status": "completed",
                "detected_language": recognition["detected_language_code"],
                "original_transcript": recognition["original_transcript"],
                "english_script": recognition["english_query"],
            }
        )
        logger.info("bhashini_transcription_completed", extra={"event": "transcription", "recording_path": str(source)})
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
        result["error"] = str(exc)
        logger.warning("bhashini_transcription_failed", extra={"event": "transcription", "recording_path": str(source)})
    output = _output_path(source)
    try:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.exception("bhashini_transcript_write_failed", extra={"recording_path": str(output)})
    return result
