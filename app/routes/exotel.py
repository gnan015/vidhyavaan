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
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.exotel import AudioFrameInfo, ExotelCallback
from app.services.audio import process_audio_frame
from app.services.recordings import download_and_process_recording, validate_recording_url
from app.services.sarvam import (
    synthesize_english_speech,
    transcribe_and_translate_audio,
    translate_wav_bytes,
    tts_wav_to_pcm16,
)
from app.services.security import verify_exotel_request

router = APIRouter(tags=["exotel"])
logger = logging.getLogger(__name__)
XML_MEDIA_TYPE = "application/xml"
STREAM_RECORDINGS_DIRECTORY = Path("recordings")
_transcription_tasks: set[asyncio.Task[dict[str, Any]]] = set()


def exotel_hangup_xml() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'


def _stream_recording_path(call_sid: str | None) -> Path:
    """Return a traversal-safe filename for an Exotel stream recording."""
    safe_call_sid = re.sub(r"[^A-Za-z0-9_.-]", "_", call_sid or "unknown")
    return STREAM_RECORDINGS_DIRECTORY / f"call_{safe_call_sid}.wav"


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
        transcribe_and_translate_audio(str(audio_path)), name="sarvam-transcription"
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
    """Wrap Exotel PCM16 in a WAV container for Sarvam's multipart API."""
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)
    return stream.getvalue()


async def _stream_pcm_to_exotel(
    websocket: WebSocket, stream_sid: str, audio: bytes, sample_rate: int
) -> None:
    """Send 100 ms PCM16 media frames at approximately real-time pace."""
    frame_size = sample_rate * 2 // 10
    for offset in range(0, len(audio), frame_size):
        chunk = audio[offset : offset + frame_size]
        await websocket.send_json(
            {
                "event": "media",
                "stream_sid": stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode("ascii")},
            }
        )
        await asyncio.sleep(0.09)


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
    call_sid: str | None = None
    stream_sid: str | None = None
    sample_rate = 8_000
    sample_width = 2
    encoding = "audio/x-raw"
    pcm_buffer = bytearray()
    audio_frames = bytearray()
    audio_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    media_chunk_count = 0
    frames_received = 0
    frames_processed = 0
    sequence_gaps = 0
    last_sequence: int | None = None
    last_chunk_size = 0
    speech_active = False
    last_speech_time: float | None = None
    is_processing = False
    utterance_task: asyncio.Task[None] | None = None

    async def process_utterance(captured_audio: bytes) -> None:
        """Translate one paused utterance and play its Sarvam TTS response."""
        nonlocal speech_active, last_speech_time, is_processing, utterance_task
        try:
            if not stream_sid:
                logger.warning(
                    "loopback_skipped_without_stream_sid",
                    extra={"call_sid": call_sid, "event": "loopback"},
                )
                return
            english_text = await translate_wav_bytes(
                _pcm_to_wav_bytes(captured_audio, sample_rate)
            )
            if not english_text:
                logger.info(
                    "loopback_empty_translation",
                    extra={"call_sid": call_sid, "event": "loopback"},
                )
                return
            tts_wav = await synthesize_english_speech(english_text, sample_rate)
            response_pcm = tts_wav_to_pcm16(tts_wav, sample_rate)
            await _stream_pcm_to_exotel(websocket, stream_sid, response_pcm, sample_rate)
            logger.info(
                "loopback_playback_completed",
                extra={"call_sid": call_sid, "event": "loopback"},
            )
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, OSError, ValueError, RuntimeError):
            logger.exception(
                "loopback_processing_failed",
                extra={"call_sid": call_sid, "event": "loopback"},
            )
        finally:
            audio_frames.clear()
            speech_active = False
            last_speech_time = None
            is_processing = False
            utterance_task = None

    async def consume_audio() -> None:
        """Decode queued frames without delaying WebSocket reads."""
        nonlocal frames_processed, sequence_gaps, last_sequence, last_chunk_size
        nonlocal speech_active, last_speech_time, is_processing, utterance_task
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
                        # lost source frames. Reuse the prior chunk duration.
                        padding_size = last_chunk_size or len(audio)
                        pcm_buffer.extend(b"\x00" * (padding_size * missing_frames))
                        sequence_gaps += missing_frames
                        logger.warning(
                            "media_sequence_gap_detected",
                            extra={"call_sid": call_sid, "event": "media", "sequence_gaps": sequence_gaps},
                        )
                if sequence is not None:
                    last_sequence = sequence
                pcm_buffer.extend(audio)
                last_chunk_size = len(audio)
                frames_processed += 1
                logger.info(
                    "exotel_media_processed",
                    extra={"call_sid": call_sid, "event": "media", "chunk_count": frames_processed},
                )
                if sample_width == 2 and not is_processing:
                    now = time.monotonic()
                    energy = _pcm_rms(audio)
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
                            is_processing = True
                            captured_audio = bytes(audio_frames)
                            utterance_task = asyncio.create_task(
                                process_utterance(captured_audio),
                                name="sarvam-exotel-loopback",
                            )
                            logger.info(
                                "loopback_utterance_detected",
                                extra={"call_sid": call_sid, "event": "loopback"},
                            )
                await process_audio_frame(
                    audio,
                    AudioFrameInfo(
                        encoding="pcm_s16le",
                        sample_rate_hz=sample_rate,
                        byte_count=len(audio),
                    ),
                    call_sid,
                )
            except Exception:
                # A bad frame or STT adapter must not abandon queued audio.
                logger.exception(
                    "audio_worker_frame_failed",
                    extra={"call_sid": call_sid, "event": "media"},
                )
            finally:
                audio_queue.task_done()

    consumer_task = asyncio.create_task(consume_audio(), name="exotel-audio-consumer")
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
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
            if event == "connected":
                logger.info("exotel_stream_connected", extra={"call_sid": call_sid, "event": "connected"})
                await websocket.send_json({"event": "ready"})
            elif event == "start":
                start = packet.get("start")
                metadata = start if isinstance(start, dict) else packet
                call_sid = (
                    metadata.get("call_sid")
                    or metadata.get("callSid")
                    or metadata.get("CallSid")
                    or call_sid
                )
                stream_sid = metadata.get("stream_sid") or metadata.get("streamSid") or stream_sid
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
                sample_rate = _sample_rate_from_media_format(media_format)
                sample_width = _sample_width_from_media_format(media_format)
                encoding = str(media_format.get("encoding", "audio/x-raw")).lower()
                if encoding == "audio/x-mulaw":
                    # Decoding produces 16-bit PCM regardless of source metadata.
                    sample_width = 2
                logger.info(
                    "exotel_stream_started",
                    extra={
                        "call_sid": call_sid,
                        "event": "start",
                        "start": metadata,
                        "media_format": media_format,
                    },
                )
            elif event == "media":
                media = packet.get("media")
                if not isinstance(media, dict) or not isinstance(media.get("payload"), str):
                    logger.warning("invalid_media_packet", extra={"call_sid": call_sid, "event": "media"})
                    continue
                frames_received += 1
                media_chunk_count += 1
                await audio_queue.put(
                    {
                        "payload": media["payload"],
                        "sequence": media.get(
                            "sequence_number",
                            media.get(
                                "chunk",
                                packet.get("sequence_number", packet.get("chunk")),
                            ),
                        ),
                    }
                )
                logger.info(
                    "exotel_media_queued",
                    extra={"call_sid": call_sid, "event": "media", "chunk_count": media_chunk_count},
                )
            elif event == "stop":
                logger.info("exotel_stream_stopped", extra={"call_sid": call_sid, "event": "stop"})
                break
    except WebSocketDisconnect:
        logger.info("exotel_stream_disconnected", extra={"call_sid": call_sid, "event": "disconnect"})
    finally:
        try:
            if utterance_task and not utterance_task.done():
                utterance_task.cancel()
                with suppress(asyncio.CancelledError):
                    await utterance_task
            await audio_queue.put(None)
            await audio_queue.join()
            await consumer_task
            audio_frames = _smooth_and_normalize_pcm(
                bytes(pcm_buffer), sample_rate, sample_width
            )
            output_path = _write_stream_wav(
                audio_frames, call_sid, sample_rate, sample_width
            )
            duration_seconds = len(audio_frames) / (sample_rate * sample_width)
            logger.info(
                "exotel_stream_wav_written",
                extra={
                    "call_sid": call_sid,
                    "event": "file_write",
                    "recording_path": str(output_path),
                    "chunk_count": media_chunk_count,
                    "stream_sid": stream_sid,
                    "total_bytes": len(audio_frames),
                    "duration_seconds": duration_seconds,
                    "frames_received": frames_received,
                    "frames_processed": frames_processed,
                    "sequence_gaps": sequence_gaps,
                },
            )
            _schedule_transcription(output_path, call_sid)
        except Exception:
            logger.exception("exotel_stream_wav_write_failed", extra={"call_sid": call_sid, "event": "file_write"})
        if not consumer_task.done():
            consumer_task.cancel()
            with suppress(asyncio.CancelledError):
                await consumer_task
        with suppress(Exception):
            await websocket.close()
