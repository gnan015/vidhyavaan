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
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.exotel import AudioFrameInfo, ExotelCallback
from app.services.audio import process_audio_frame
from app.services.recordings import download_and_process_recording, validate_recording_url
from app.services.sarvam import (
    synthesize_english_speech,
    transcribe_and_translate_audio,
    translate_wav_to_english,
    tts_wav_to_pcm16,
)
from app.services.rag_middleware import query_rag
from app.services.security import verify_exotel_request

router = APIRouter(tags=["exotel"])
logger = logging.getLogger(__name__)
XML_MEDIA_TYPE = "application/xml"
STREAM_RECORDINGS_DIRECTORY = Path("recordings")
_transcription_tasks: set[asyncio.Task[dict[str, Any]]] = set()
_FALLBACK_RESPONSES = {
    "en-IN": "Sorry, I could not find information on that.",
    "hi-IN": "माफ़ कीजिए, मुझे इस बारे में जानकारी नहीं मिली।",
    "te-IN": "క్షమించండి, దీని గురించి నాకు సమాచారం దొరకలేదు.",
    "ta-IN": "மன்னிக்கவும், இதைப் பற்றிய தகவல் கிடைக்கவில்லை.",
    "kn-IN": "ಕ್ಷಮಿಸಿ, ಇದರ ಬಗ್ಗೆ ಮಾಹಿತಿ ಸಿಗಲಿಲ್ಲ.",
}


def exotel_hangup_xml() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'


def _stream_recording_path(call_sid: str | None) -> Path:
    """Return a traversal-safe filename for an Exotel stream recording."""
    safe_call_sid = re.sub(r"[^A-Za-z0-9_.-]", "_", call_sid or "unknown")
    return STREAM_RECORDINGS_DIRECTORY / f"call_{safe_call_sid}.wav"


def _fallback_response(language_code: str) -> str:
    # Keep recovery speech conversational as well; it must not pass through a
    # literal English-to-Telugu translation that would sound formal on a call.
    if language_code.lower().startswith("te"):
        return "Sorry, ippudu ee question ki answer dorakaledu. Please malli adagandi."
    return _FALLBACK_RESPONSES.get(language_code, _FALLBACK_RESPONSES["en-IN"])


def _shorten_for_voice(text: str, limit: int = 420) -> str:
    """Keep spoken answers brief enough for reliable telephony TTS requests."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    boundary = max(
        normalized.rfind(mark, 0, limit)
        for mark in (". ", "? ", "! ", "। ")
    )
    if boundary > limit // 2:
        return normalized[: boundary + 1]
    word_boundary = normalized.rfind(" ", 0, limit)
    return normalized[: word_boundary if word_boundary > 0 else limit].rstrip() + "..."


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


def _turn_telemetry(message: str, *, call_sid: str | None = None) -> None:
    """Print concise turn telemetry and retain it in structured application logs."""
    print(message, flush=True)
    logger.info(message, extra={"call_sid": call_sid, "event": "loopback"})


def _stage_error(stage: str, error: Exception, *, call_sid: str | None) -> None:
    """Log a recoverable turn-stage failure without ending the WebSocket call."""
    message = f"[ERROR AT STAGE: {stage}] {type(error).__name__}: {error}"
    print(message, flush=True)
    logger.exception(
        "loopback_stage_failed",
        extra={"call_sid": call_sid, "event": "loopback", "stage": stage},
    )
    # Keep a complete traceback in the Uvicorn console for live diagnosis.
    traceback.print_exc()


class _ExotelOutboundSender:
    """Serialize outbound media so keepalive and TTS frames cannot interleave."""

    def __init__(self, websocket: WebSocket, stream_sid: str) -> None:
        self._websocket = websocket
        self._stream_sid = stream_sid
        self._lock = asyncio.Lock()

    async def send_pcm(self, audio: bytes, sample_rate: int) -> None:
        """Send Exotel's minimal documented bidirectional media envelope."""
        async with self._lock:
            await self._websocket.send_json(
                {
                    "event": "media",
                    "stream_sid": self._stream_sid,
                    "media": {
                        "payload": base64.b64encode(audio).decode("ascii"),
                    },
                }
            )


def _exotel_frame_size(sample_rate: int) -> int:
    """Return a 320-byte-aligned Exotel payload size, at least 3,200 bytes."""
    return max(sample_rate * 2 // 10, 3_200) // 320 * 320


async def _stream_pcm_to_exotel(
    websocket: WebSocket,
    stream_sid: str,
    audio: bytes,
    sample_rate: int,
    sender: _ExotelOutboundSender | None = None,
) -> int:
    """Send Exotel-compliant PCM16 frames at approximately real-time pace.

    Exotel requires each bidirectional payload to be at least 3,200 bytes and
    a multiple of 320 bytes.  At the required 8 kHz PCM playback rate this is
    a 200 ms frame (rather than the usual 100 ms/1,600 byte PCM frame).
    """
    frame_size = _exotel_frame_size(sample_rate)
    frame_duration_seconds = frame_size / (sample_rate * 2)
    frames_sent = 0
    sender = sender or _ExotelOutboundSender(websocket, stream_sid)
    for offset in range(0, len(audio), frame_size):
        chunk = audio[offset : offset + frame_size]
        # Exotel rejects a short final frame; pad it with playback silence.
        if len(chunk) < frame_size:
            chunk = chunk.ljust(frame_size, b"\x00")
        await sender.send_pcm(chunk, sample_rate)
        frames_sent += 1
        # Leave a small scheduling margin while preserving real-time playback.
        await asyncio.sleep(max(0.0, frame_duration_seconds - 0.01))
    return frames_sent


async def _stream_processing_keepalive(
    sender: _ExotelOutboundSender,
    sample_rate: int,
    stop_event: asyncio.Event,
    call_sid: str | None,
    *,
    first_frame_already_sent: bool = False,
) -> None:
    """Keep the Voicebot stream active while STT/RAG/TTS waits on APIs."""
    frame_size = _exotel_frame_size(sample_rate)
    silence_frame = b"\x00" * frame_size
    interval_seconds = frame_size / (sample_rate * 2)
    sent = 0
    _turn_telemetry("[KEEPALIVE] Streaming silent audio while the answer is prepared.", call_sid=call_sid)
    try:
        # The caller sends the first frame synchronously before STT starts.  Do
        # not queue a duplicate 400 ms of silence at the beginning of a turn.
        if first_frame_already_sent:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except TimeoutError:
                pass
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
    finally:
        _turn_telemetry(f"[KEEPALIVE] Stopped after {sent} silent frame(s).", call_sid=call_sid)


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
    is_bot_turn = False
    utterance_task: asyncio.Task[None] | None = None
    outbound_sender: _ExotelOutboundSender | None = None

    async def process_utterance(captured_audio: bytes) -> None:
        """Run one recoverable STT -> RAG -> TTS turn without idling Exotel."""
        nonlocal speech_active, last_speech_time, is_bot_turn, utterance_task
        playback_sample_rate = settings.exotel_playback_sample_rate or sample_rate
        keepalive_stop = asyncio.Event()
        keepalive_task: asyncio.Task[None] | None = None

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
                    "[TTS START] Synthesizing the recovery response via Sarvam AI...",
                    call_sid=call_sid,
                )
                fallback_started = time.perf_counter()
                fallback_wav = await synthesize_english_speech(
                    fallback_text,
                    playback_sample_rate,
                    language_code,
                )
                fallback_pcm = tts_wav_to_pcm16(
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
                    first_frame_already_sent=True,
                ),
                name="exotel-processing-keepalive",
            )
            # Give the keepalive task one scheduling turn before the STT worker
            # starts its network request.
            await asyncio.sleep(0)

            try:
                wav_audio = _pcm_to_wav_bytes(captured_audio, sample_rate)
                _turn_telemetry(
                    f"[STT START] Sending {len(wav_audio):,} bytes to Sarvam AI STT...",
                    call_sid=call_sid,
                )
                stt_started = time.perf_counter()
                recognition = await translate_wav_to_english(wav_audio)
                english_query = str(recognition.get("english_query") or "").strip()
                detected_language_code = str(
                    recognition.get("detected_language_code") or "en-IN"
                )
                _turn_telemetry(
                    "[STT FINISH] Transcribed in "
                    f"{time.perf_counter() - stt_started:.2f}s | Detected: "
                    f"'{detected_language_code}' | Query: '{english_query[:500]}'",
                    call_sid=call_sid,
                )
                if not english_query:
                    raise ValueError("Sarvam returned an empty transcription")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _stage_error("STT", exc, call_sid=call_sid)
                await play_failure_message()
                return

            if settings.live_rag_enabled:
                try:
                    _turn_telemetry(
                        "[RAG START] Querying middleware knowledge base...", call_sid=call_sid
                    )
                    rag_started = time.perf_counter()
                    # query_rag is asynchronous and already moves the synchronous
                    # PDF/Groq middleware into asyncio.to_thread internally.
                    english_answer = await query_rag(
                        english_query,
                        call_sid or stream_sid or "anonymous-call",
                        detected_language_code,
                    )
                    if not english_answer:
                        raise RuntimeError("RAG middleware returned no usable answer")
                    _turn_telemetry(
                        "[RAG FINISH] RAG responded in "
                        f"{time.perf_counter() - rag_started:.2f}s | Answer: "
                        f"'{english_answer[:500]}'",
                        call_sid=call_sid,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _stage_error("RAG", exc, call_sid=call_sid)
                    await play_failure_message(detected_language_code)
                    return
            else:
                # This is only used when LIVE_RAG_ENABLED is deliberately off.
                # It does not invoke mechanical full-text translation.
                english_answer = english_query
                _turn_telemetry(
                    "[RAG SKIPPED] Direct response mode is active.",
                    call_sid=call_sid,
                )

            # RAG/Groq already generated the answer directly in the target voice
            # style (for example, Tinglish). Keep it brief for TTS, but never
            # run a full-text English-to-native-language translation afterwards.
            spoken_answer = _shorten_for_voice(english_answer, limit=360)

            try:
                _turn_telemetry(
                    f"[TTS INPUT] {len(spoken_answer)} characters in direct voice style.",
                    call_sid=call_sid,
                )
                _turn_telemetry(
                    "[TTS START] Synthesizing speech via Sarvam AI...", call_sid=call_sid
                )
                tts_started = time.perf_counter()
                tts_wav = await synthesize_english_speech(
                    spoken_answer, playback_sample_rate, detected_language_code
                )
                response_pcm = tts_wav_to_pcm16(tts_wav, playback_sample_rate)
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
                await play_failure_message(detected_language_code)
                return

            try:
                await stop_keepalive()
                frame_size = _exotel_frame_size(playback_sample_rate)
                expected_frames = (len(response_pcm) + frame_size - 1) // frame_size
                _turn_telemetry(
                    f"[PLAYBACK] Streaming {expected_frames} frame(s) back to Exotel...",
                    call_sid=call_sid,
                )
                playback_started = time.perf_counter()
                frames_sent = await _stream_pcm_to_exotel(
                    websocket,
                    stream_sid,
                    response_pcm,
                    playback_sample_rate,
                    outbound_sender,
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
            audio_frames.clear()
            speech_active = False
            last_speech_time = None
            is_bot_turn = False
            utterance_task = None

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
                if sample_width == 2 and not is_bot_turn:
                    try:
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
                                is_bot_turn = True
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
            if event == "connected":
                logger.info("exotel_stream_connected", extra={"call_sid": call_sid, "event": "connected"})
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
                if stream_sid:
                    outbound_sender = _ExotelOutboundSender(websocket, stream_sid)
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
                if is_bot_turn:
                    logger.debug(
                        "caller_audio_discarded_during_bot_turn",
                        extra={"call_sid": call_sid, "event": "loopback"},
                    )
                    continue
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
