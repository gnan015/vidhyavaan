"""
test_sarvam_audio.py
====================
Standalone test script for Sarvam AI's Speech-to-Text (STT) and
Text-to-Speech (TTS) pipelines targeting Indian regional languages.

Usage:
    python test_sarvam_audio.py

Requirements:
    pip install requests

Replace SARVAM_API_KEY with your actual key before running.
"""

import base64
import os
import sys

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SARVAM_API_KEY: str = "YOUR_SARVAM_API_KEY"  # <-- replace with your key

STT_URL = "https://api.sarvam.ai/speech-to-text"
TTS_URL = "https://api.sarvam.ai/text-to-speech"

# ---------------------------------------------------------------------------
# Speech-to-Text (STT)
# ---------------------------------------------------------------------------


def speech_to_text(audio_file_path: str, language_code: str = "te-IN") -> str:
    """
    Convert a local audio file to text using Sarvam AI's STT API.

    Args:
        audio_file_path: Path to a WAV audio file on disk.
        language_code:   BCP-47 language tag, e.g. "te-IN" (Telugu),
                         "hi-IN" (Hindi), "kn-IN" (Kannada).

    Returns:
        The transcribed text string returned by Sarvam AI.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        requests.HTTPError: If the API returns a non-2xx status.
        KeyError: If the response JSON does not contain a "transcript" key.
    """
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
    }

    # The STT endpoint expects multipart/form-data with the audio binary and
    # a JSON-encoded 'model' field.
    with open(audio_file_path, "rb") as audio_file:
        files = {
            "file": (os.path.basename(audio_file_path), audio_file, "audio/wav"),
        }
        data = {
            "model": "saaras:v2",
            "language_code": language_code,
        }

        print(f"[STT] Sending '{audio_file_path}' to Sarvam AI ...")
        response = requests.post(
            STT_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=60,
        )

    # Raise immediately on HTTP errors so callers get a clear exception.
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"[STT] HTTP error {response.status_code}: {response.text}")
        raise exc

    payload = response.json()

    # The Sarvam STT response contains a top-level "transcript" key.
    if "transcript" not in payload:
        raise KeyError(
            f"[STT] Unexpected response format — 'transcript' key missing. "
            f"Full response: {payload}"
        )

    transcript: str = payload["transcript"]
    print(f"[STT] Transcript: {transcript!r}")
    return transcript


# ---------------------------------------------------------------------------
# Text-to-Speech (TTS)
# ---------------------------------------------------------------------------


def text_to_speech(
    text: str,
    language_code: str = "te-IN",
    output_filename: str = "output.wav",
) -> bytes:
    """
    Convert text to speech using Sarvam AI's TTS API and save it as a WAV file.

    Args:
        text:            The text to synthesize (plain string).
        language_code:   BCP-47 language tag, e.g. "te-IN" (Telugu).
        output_filename: Local file path where the WAV will be written.

    Returns:
        Raw WAV audio bytes.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
        KeyError:           If the response JSON does not contain expected keys.
        ValueError:         If the base64-encoded audio payload cannot be decoded.
    """
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    body = {
        "inputs": [text],
        "target_language_code": language_code,
        "speaker": "meera",
        "model": "bulbul:v1",
        "speech_sample_rate": 8000,    # 8 kHz — telephony standard, matches Exotel
        "enable_preprocessing": True,  # normalise numbers, abbreviations, etc.
        "pitch": 0,                    # neutral pitch (range: -0.75 to 0.75)
        "pace": 1.0,                   # normal speed
        "loudness": 1.0,               # normal loudness
    }

    print(f"[TTS] Synthesising speech for language '{language_code}' ...")
    response = requests.post(
        TTS_URL,
        headers=headers,
        json=body,
        timeout=60,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"[TTS] HTTP error {response.status_code}: {response.text}")
        raise exc

    payload = response.json()

    # Sarvam TTS returns: { "audios": ["<base64-wav>", ...] }
    if "audios" not in payload or not payload["audios"]:
        raise KeyError(
            f"[TTS] Unexpected response format — 'audios' key missing or empty. "
            f"Full response: {payload}"
        )

    audio_b64: str = payload["audios"][0]

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as exc:
        raise ValueError(
            "[TTS] Failed to base64-decode the audio payload."
        ) from exc

    # Persist the WAV file locally.
    with open(output_filename, "wb") as out_file:
        out_file.write(audio_bytes)

    print(f"[TTS] Audio saved → '{output_filename}' ({len(audio_bytes):,} bytes)")
    return audio_bytes


# ---------------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Configuration check — warn early if the placeholder key is still set.
    # ------------------------------------------------------------------
    if SARVAM_API_KEY == "YOUR_SARVAM_API_KEY":
        print(
            "[ERROR] SARVAM_API_KEY is still set to the placeholder value.\n"
            "        Edit test_sarvam_audio.py and replace it with your real key."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Speech-to-Text
    # ------------------------------------------------------------------
    input_audio_path = "test_input.wav"

    if not os.path.isfile(input_audio_path):
        print(
            f"[ERROR] Input audio file '{input_audio_path}' not found.\n"
            f"        Place a WAV file named 'test_input.wav' next to this script."
        )
        sys.exit(1)

    print("=" * 60)
    print(" STEP 1 — Speech-to-Text")
    print("=" * 60)

    try:
        transcript = speech_to_text(
            audio_file_path=input_audio_path,
            language_code="te-IN",   # Telugu; change as needed
        )
    except Exception as exc:
        print(f"[FATAL] STT failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Build a response string from the transcript.
    # ------------------------------------------------------------------
    # In a real pipeline this is where your NLP / LLM / dialogue manager
    # would produce a contextual reply. Here we append a simple echo prefix
    # so that the TTS step has something meaningful to synthesise.
    response_text = f"మీరు చెప్పారు: {transcript}"   # "You said: <transcript>" in Telugu
    print(f"\n[APP]  Response text: {response_text!r}")

    # ------------------------------------------------------------------
    # Step 3: Text-to-Speech
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(" STEP 2 — Text-to-Speech")
    print("=" * 60)

    output_path = "output_response.wav"

    try:
        audio_bytes = text_to_speech(
            text=response_text,
            language_code="te-IN",
            output_filename=output_path,
        )
    except Exception as exc:
        print(f"[FATAL] TTS failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(" DONE")
    print("=" * 60)
    print(f"  Input audio  : {input_audio_path}")
    print(f"  Transcript   : {transcript!r}")
    print(f"  Response text: {response_text!r}")
    print(f"  Output audio : {output_path}  ({len(audio_bytes):,} bytes)")
    print()
    print("Play the output file with any WAV-compatible media player.")
