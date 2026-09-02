"""
process_recent_audio.py
=======================
Production-grade Sarvam AI audio pipeline.

Pipeline:
  1. Scan ./inputs/ for the most recently modified audio file.
  2. Convert non-WAV files to 8 kHz WAV (via pydub) in-memory.
  3. Send audio to Sarvam STT  → detect language + get English transcript.
  4. Send transcript to Sarvam TTS (en-IN) → save timestamped WAV in ./outputs/.

Usage:
  python process_recent_audio.py

Requirements (install via pip or see requirements_audio.txt):
  pip install requests tenacity pydub
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Logging setup — clean, timestamp-prefixed terminal output
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("sarvam_pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Read from environment; fall back to the placeholder below for quick local tests.
SARVAM_API_KEY_FALLBACK: str = "sk_4r5w9e4b_I7ApxAGis80e61wGOlocjL0H"  # <-- replace or set env var
SARVAM_API_KEY: str = os.environ.get("SARVAM_API_KEY", SARVAM_API_KEY_FALLBACK)

STT_URL: str = "https://api.sarvam.ai/speech-to-text"
TTS_URL: str = "https://api.sarvam.ai/text-to-speech"

INPUTS_DIR: Path = Path("./inputs")
OUTPUTS_DIR: Path = Path("./outputs")

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".ogg")

# STT settings
STT_MODEL: str = "saaras:v4"   # latest stable Sarvam STT model
STT_MODE: str = "translate"    # translate mode → always returns English + enables language detection

# TTS settings
TTS_MODEL: str = "bulbul:v3"
TTS_SPEAKER: str = "aditya"    # valid for bulbul:v3: aditya, ritu, ashutosh, priya, neha, rahul, pooja ...
TTS_LANGUAGE: str = "en-IN"    # Indian English output
TTS_SAMPLE_RATE: int = 8000    # 8 kHz — telephony / Exotel standard

# Retry policy (applied to every HTTP call)
_RETRY_ATTEMPTS: int = 3
_RETRY_MIN_WAIT: float = 1.0   # seconds
_RETRY_MAX_WAIT: float = 8.0   # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    """Create ./inputs and ./outputs directories if they do not already exist."""
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _find_most_recent_audio() -> Path:
    """
    Scan INPUTS_DIR and return the most recently modified supported audio file.

    Raises:
        SystemExit: If the folder is empty or contains no supported audio files.
    """
    candidates = [
        p for p in INPUTS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not candidates:
        log.warning(
            "No supported audio files found in '%s'.\n"
            "  Drop a .wav / .mp3 / .m4a / .ogg file there and re-run.",
            INPUTS_DIR.resolve(),
        )
        sys.exit(0)

    # Sort by modification time descending — pick newest first.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]

    log.info(
        "Found %d audio file(s) in '%s'.",
        len(candidates),
        INPUTS_DIR,
    )
    log.info(
        "Selected : %s  (%.1f KB, modified %s)",
        chosen.name,
        chosen.stat().st_size / 1024,
        datetime.fromtimestamp(chosen.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    )
    return chosen


def _to_wav_bytes(audio_path: Path, target_sample_rate: int = 8000) -> bytes:
    """
    Return the audio file as raw WAV bytes at `target_sample_rate`.

    - If the file is already a .wav it is read directly.
    - All other formats (mp3, m4a, ogg) are decoded via pydub and
      re-encoded as mono 16-bit PCM WAV at `target_sample_rate`.

    Args:
        audio_path:         Path to the source audio file.
        target_sample_rate: Output sample rate in Hz (default 8000).

    Returns:
        Raw WAV bytes ready for upload.

    Raises:
        ImportError: If pydub is not installed for non-WAV files.
        RuntimeError: If pydub fails to decode the file.
    """
    suffix = audio_path.suffix.lower()

    if suffix == ".wav":
        log.info("WAV file detected — reading directly.")
        return audio_path.read_bytes()

    # Non-WAV path — requires pydub + ffmpeg
    log.info("Converting %s → WAV @ %d Hz via pydub ...", suffix, target_sample_rate)
    try:
        from pydub import AudioSegment  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pydub is required to convert non-WAV files. "
            "Install it with: pip install pydub\n"
            "You also need ffmpeg on your PATH: https://ffmpeg.org/download.html"
        ) from exc

    fmt_map = {".mp3": "mp3", ".m4a": "mp4", ".ogg": "ogg"}
    fmt = fmt_map.get(suffix, suffix.lstrip("."))

    try:
        segment = AudioSegment.from_file(str(audio_path), format=fmt)
    except Exception as exc:
        raise RuntimeError(
            f"pydub could not decode '{audio_path.name}': {exc}\n"
            "Ensure ffmpeg is installed and accessible in your PATH."
        ) from exc

    # Normalise: mono, 16-bit, target sample rate
    segment = segment.set_channels(1).set_sample_width(2).set_frame_rate(target_sample_rate)

    buf = io.BytesIO()
    segment.export(buf, format="wav")
    wav_bytes = buf.getvalue()

    log.info("Conversion complete — WAV buffer: %.1f KB", len(wav_bytes) / 1024)
    return wav_bytes


# ---------------------------------------------------------------------------
# Retry decorator — wraps every API call
# ---------------------------------------------------------------------------


def _api_retry(func):  # type: ignore[no-untyped-def]
    """
    Decorator: retry on transient HTTP / connection errors with exponential backoff.
    Attempts: 3   |   Back-off: 1 s -> 2 s -> 4 s (capped at 8 s).
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )(func)


# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------


@_api_retry
def _stt_request(wav_bytes: bytes, filename: str) -> dict:
    """
    POST wav_bytes to Sarvam STT.  Returns the parsed JSON response dict.

    Uses 'translate' mode so output language is always English and the
    API also returns the detected source language.
    """
    headers = {"api-subscription-key": SARVAM_API_KEY}
    files = {"file": (filename, wav_bytes, "audio/wav")}
    data = {
        "model": STT_MODEL,
        "with_timestamps": False,
        "mode": STT_MODE,  # 'translate' => English text + language detection
    }

    log.info("[STT] Sending audio to Sarvam AI ...")
    response = requests.post(STT_URL, headers=headers, files=files, data=data, timeout=60)

    try:
        response.raise_for_status()
    except requests.HTTPError:
        log.error("[STT] HTTP %d: %s", response.status_code, response.text[:300])
        raise

    return response.json()


def speech_to_text(audio_path: Path) -> tuple[str, str]:
    """
    Run the full STT pipeline for `audio_path`.

    Returns:
        (detected_language, english_transcript)
        detected_language is the BCP-47 tag returned by Sarvam (e.g. "te-IN").
        Falls back to "unknown" if the API does not provide it.
    """
    wav_bytes = _to_wav_bytes(audio_path, target_sample_rate=TTS_SAMPLE_RATE)

    try:
        payload = _stt_request(wav_bytes, filename=audio_path.stem + ".wav")
    except RetryError as exc:
        raise RuntimeError("[STT] All retry attempts exhausted.") from exc

    # --- Parse transcript ---
    transcript: str = payload.get("transcript", "")
    if not transcript:
        raise KeyError(
            f"[STT] 'transcript' key missing or empty in response: {payload}"
        )

    # --- Parse detected language ---
    # Sarvam returns language_code at top-level or inside a 'language_identification' block.
    detected_lang: str = (
        payload.get("language_code")
        or payload.get("detected_language")
        or (payload.get("language_identification") or {}).get("language_code")
        or "unknown"
    )

    log.info("[STT] Detected language : %s", detected_lang)
    log.info("[STT] English transcript: %r", transcript)

    return detected_lang, transcript


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------


@_api_retry
def _tts_request(text: str) -> dict:
    """POST text to Sarvam TTS.  Returns the parsed JSON response dict."""
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "inputs": [text],
        "target_language_code": TTS_LANGUAGE,
        "speaker": TTS_SPEAKER,
        "model": TTS_MODEL,
        "speech_sample_rate": TTS_SAMPLE_RATE,
        "enable_preprocessing": True,
        "pace": 1.0,
    }

    log.info("[TTS] Sending text to Sarvam AI ...")
    response = requests.post(TTS_URL, headers=headers, json=body, timeout=60)

    try:
        response.raise_for_status()
    except requests.HTTPError:
        log.error("[TTS] HTTP %d: %s", response.status_code, response.text[:300])
        raise

    return response.json()


def text_to_speech(text: str) -> Path:
    """
    Run the full TTS pipeline for `text`.

    Saves the synthesised WAV to OUTPUTS_DIR/output_<timestamp>.wav.

    Returns:
        Path to the saved WAV file.
    """
    try:
        payload = _tts_request(text)
    except RetryError as exc:
        raise RuntimeError("[TTS] All retry attempts exhausted.") from exc

    audios: list = payload.get("audios", [])
    if not audios:
        raise KeyError(
            f"[TTS] 'audios' key missing or empty in response: {payload}"
        )

    try:
        audio_bytes = base64.b64decode(audios[0])
    except Exception as exc:
        raise ValueError("[TTS] Failed to base64-decode the audio payload.") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"output_{timestamp}.wav"
    output_path.write_bytes(audio_bytes)

    log.info(
        "[TTS] Audio saved -> '%s'  (%.1f KB)",
        output_path,
        len(audio_bytes) / 1024,
    )
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full pipeline: File picker -> STT -> TTS -> Save."""

    # ------------------------------------------------------------------
    # Guard: API key must not be the placeholder
    # ------------------------------------------------------------------
    if not SARVAM_API_KEY or SARVAM_API_KEY == "YOUR_SARVAM_API_KEY":
        log.error(
            "SARVAM_API_KEY is not set.\n"
            "  Option 1: set env var:  export SARVAM_API_KEY=sk_...\n"
            "  Option 2: edit SARVAM_API_KEY_FALLBACK inside this script."
        )
        sys.exit(1)

    _ensure_dirs()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("  Sarvam AI Audio Pipeline")
    log.info("  STT model : %s  (mode: %s)", STT_MODEL, STT_MODE)
    log.info("  TTS model : %s  |  speaker: %s  |  lang: %s", TTS_MODEL, TTS_SPEAKER, TTS_LANGUAGE)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Pick most recent audio file
    # ------------------------------------------------------------------
    audio_path = _find_most_recent_audio()

    # ------------------------------------------------------------------
    # Step 2 — Speech-to-Text (any Indian language -> English)
    # ------------------------------------------------------------------
    log.info("")
    log.info("-- STEP 1 -- Speech-to-Text ---------------------------------")

    try:
        detected_lang, english_transcript = speech_to_text(audio_path)
    except Exception as exc:
        log.error("FATAL — STT stage failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3 — Text-to-Speech (English -> Indian English audio)
    # ------------------------------------------------------------------
    log.info("")
    log.info("-- STEP 2 -- Text-to-Speech ---------------------------------")

    try:
        output_wav = text_to_speech(english_transcript)
    except Exception as exc:
        log.error("FATAL — TTS stage failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("  Input file        : %s", audio_path.resolve())
    log.info("  Detected language : %s", detected_lang)
    log.info("  English text      : %s", english_transcript)
    log.info("  Output audio      : %s", output_wav.resolve())
    log.info("=" * 60)
    log.info("  Play the output with any WAV-compatible media player.")
    log.info("")


if __name__ == "__main__":
    main()
