"""Sarvam STT and English-translation post-processing for call recordings."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


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

        timeout = httpx.Timeout(settings.sarvam_request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
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
