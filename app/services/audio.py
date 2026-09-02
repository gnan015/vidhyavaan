import base64
import binascii
import logging

from app.schemas.exotel import AudioFrameInfo

logger = logging.getLogger(__name__)
SUPPORTED_ENCODINGS = {"mulaw", "mu-law", "pcm", "pcm_s16le", "linear16"}


def decode_audio_payload(payload: dict) -> tuple[bytes, AudioFrameInfo]:
    """Decode a typical JSON media packet with base64 `payload` audio."""
    raw_payload = payload.get("payload") or payload.get("audio")
    if not isinstance(raw_payload, str):
        raise ValueError("Audio media packet must contain base64 payload or audio")
    try:
        audio = base64.b64decode(raw_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Audio payload is not valid base64") from exc

    encoding = str(payload.get("encoding", "mulaw")).lower()
    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(f"Unsupported audio encoding: {encoding}")
    rate = int(payload.get("sample_rate", payload.get("sampleRate", 8000)))
    if rate not in {8000, 16000}:
        raise ValueError("Only 8 kHz and 16 kHz audio is supported")
    return audio, AudioFrameInfo(encoding=encoding, sample_rate_hz=rate, byte_count=len(audio))


async def process_audio_frame(audio: bytes, frame: AudioFrameInfo, call_sid: str | None) -> None:
    """STT integration point. Queue `audio` to your speech engine here."""
    logger.debug("audio_frame_received", extra={"call_sid": call_sid, "event": "audio_frame"})

