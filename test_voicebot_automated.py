"""End-to-end mock-Exotel test suite for Telugu, Tamil, and Bengali calls.

This uses real Bhashini and Groq API calls. Configure Bhashini credentials and
``GROQ_API_KEY`` in ``.env``; no phone number, ngrok tunnel, or Exotel account
is required.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import re
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


import httpx
import websockets

from app.core.config import get_settings
from app.services.bhashini import synthesize_speech, tts_audio_to_pcm16


PROJECT_DIRECTORY = Path(__file__).resolve().parent
RECORDINGS_DIRECTORY = PROJECT_DIRECTORY / "recordings"
SAMPLE_RATE, SAMPLE_WIDTH = 8_000, 2
FRAME_SIZE = SAMPLE_RATE * SAMPLE_WIDTH // 10  # 100 ms PCM16 mono.
SILENCE_FRAME_COUNT = 12  # 1.2 seconds; VAD triggers at 0.9 seconds.


@dataclass(frozen=True)
class VoicebotCase:
    language_code: str
    question: str
    technical_terms: tuple[str, ...]

    @property
    def input_path(self) -> Path:
        return RECORDINGS_DIRECTORY / f"test_input_{self.language_code}.wav"

    @property
    def output_path(self) -> Path:
        return RECORDINGS_DIRECTORY / f"test_output_{self.language_code}.wav"

    @property
    def call_sid(self) -> str:
        return f"test_call_{self.language_code.replace('-', '_')}"

    @property
    def stream_sid(self) -> str:
        return f"mock_stream_{self.language_code.replace('-', '_')}"


TEST_CASES = (
    VoicebotCase(
        "te-IN",
        "Operating system lo scheduling algorithms ante enti?",
        ("CPU", "Scheduling", "Round Robin", "Algorithm"),
    ),
    VoicebotCase(
        "ta-IN",
        "Operating system-la deadlock na enna, adha epdi handle pannuvanga?",
        ("Deadlock", "Process", "Resource", "Operating System"),
    ),
    VoicebotCase(
        "bn-IN",
        "Operating system e virtual memory ki bhabe kaaj kore?",
        ("Virtual Memory", "RAM", "Paging"),
    ),
)


@dataclass
class TestReport:
    response_pcm: bytearray = field(default_factory=bytearray)
    server_logs: list[str] = field(default_factory=list)
    outbound_frames: int = 0
    non_silent_frames: int = 0
    first_outbound_at: float | None = None
    first_audible_at: float | None = None


def write_pcm_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


async def create_input(case: VoicebotCase) -> bytes:
    """Synthesize one caller WAV and return its 8 kHz PCM frames."""
    wav_data = await synthesize_speech(case.question, SAMPLE_RATE, case.language_code)
    pcm_data = tts_audio_to_pcm16(wav_data, SAMPLE_RATE)
    if not pcm_data:
        raise RuntimeError(f"Bhashini generated empty input audio for {case.language_code}")
    write_pcm_wav(case.input_path, pcm_data)
    return pcm_data


async def capture_server_output(
    process: subprocess.Popen[str], active_report: list[TestReport | None]
) -> None:
    """Mirror Uvicorn telemetry into the currently executing test report."""
    assert process.stdout is not None
    while line := await asyncio.to_thread(process.stdout.readline):
        print(f"[server] {line}", end="")
        if active_report[0] is not None:
            active_report[0].server_logs.append(line.rstrip())


async def wait_for_health(base_url: str, process: subprocess.Popen[str], timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=1.0) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Uvicorn exited before the health check completed")
            try:
                if (await client.get(f"{base_url}/health")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
    raise TimeoutError("Timed out waiting for the local FastAPI server")


def packet(event: str, **values: Any) -> str:
    return json.dumps({"event": event, **values})


async def receive_outbound_media(websocket: Any, report: TestReport) -> None:
    """Collect audible media and retain timing for the first return packet."""
    try:
        while True:
            message = json.loads(await websocket.recv())
            if message.get("event") != "media":
                continue
            media = message.get("media")
            payload = media.get("payload") if isinstance(media, dict) else None
            if not isinstance(payload, str):
                continue
            pcm = base64.b64decode(payload, validate=True)
            now = time.monotonic()
            report.outbound_frames += 1
            report.first_outbound_at = report.first_outbound_at or now
            if any(pcm):
                report.response_pcm.extend(pcm)
                report.non_silent_frames += 1
                report.first_audible_at = report.first_audible_at or now
    except asyncio.CancelledError:
        raise
    except websockets.exceptions.ConnectionClosed:
        return


async def wait_for_log(report: TestReport, markers: tuple[str, ...], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(marker in line for marker in markers for line in report.server_logs):
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {markers!r}. Recent logs:\n" + "\n".join(report.server_logs[-30:]))


def value_from_logs(report: TestReport, pattern: str, default: str = "not available") -> str:
    match: re.Match[str] | None = None
    for line in report.server_logs:
        if found := re.search(pattern, line):
            match = found
    return match.group(1) if match else default


def report_case(case: VoicebotCase, report: TestReport, silence_started_at: float) -> None:
    detected_language = value_from_logs(report, r"Detected: '([^']+)'")
    detected_question = value_from_logs(report, r"Query: '([^']*)'")
    response_text = value_from_logs(report, r"Answer: '([^']*)'")
    route = "Groq fallback" if any("[GROQ FALLBACK START]" in line for line in report.server_logs) else "RAG"
    first_audio_at = report.first_audible_at or report.first_outbound_at
    latency = first_audio_at - silence_started_at if first_audio_at else None
    is_roman = response_text.isascii()
    terms = [term for term in case.technical_terms if term.lower() in response_text.lower()]

    print(f"\n=== {case.language_code} voicebot report ===")
    print(f"Input Audio: {case.input_path} ({wav_duration_seconds(case.input_path):.2f}s)")
    settings = get_settings()
    script_mode = getattr(settings, "bhashini_tts_script_mode", "native")
    print(f"Detected Language: {detected_language}")
    print(f"Detected Question: {detected_question}")
    print(f"Query Route: {route}")
    print(f"Response Text: {response_text}")
    print(f"Script Mode: {script_mode}")
    print(f"Script Format Check: {'Roman transliterated (ASCII)' if is_roman else 'Native Indic Unicode'}")
    print(f"English technical terms retained: {', '.join(terms) or 'none found'}")
    print(f"Output Audio: {case.output_path} ({case.output_path.stat().st_size:,} bytes, {wav_duration_seconds(case.output_path):.2f}s)")
    print(f"Turnaround Latency: {latency:.2f}s" if latency is not None else "Turnaround Latency: not available")

    if detected_language != case.language_code:
        raise AssertionError(f"Expected language {case.language_code}, received {detected_language}")
    if script_mode == "roman":
        if not is_roman:
            raise AssertionError("Expected a Roman-transliterated answer")
    else:
        if not response_text.strip():
            raise AssertionError("Expected a non-empty native script answer")
    if not terms:
        raise AssertionError(f"No expected English technical term was retained: {case.technical_terms}")


async def run_case(
    case: VoicebotCase, websocket_url: str, input_pcm: bytes, active_report: list[TestReport | None]
) -> None:
    report = TestReport()
    active_report[0] = report
    try:
        async with websockets.connect(websocket_url, max_size=None) as websocket:
            receiver_task = asyncio.create_task(receive_outbound_media(websocket, report))
            try:
                await websocket.send(packet("start", stream_sid=case.stream_sid, start={"call_sid": case.call_sid, "media_format": {"encoding": "audio/x-raw", "sample_rate": SAMPLE_RATE, "bit_rate": "16", "channels": 1}}))
                sequence = 1
                for offset in range(0, len(input_pcm), FRAME_SIZE):
                    chunk = input_pcm[offset : offset + FRAME_SIZE].ljust(FRAME_SIZE, b"\x00")
                    await websocket.send(packet("media", stream_sid=case.stream_sid, media={"payload": base64.b64encode(chunk).decode("ascii"), "sequence_number": sequence}))
                    sequence += 1
                    await asyncio.sleep(0.095)

                silence_started_at = time.monotonic()
                silence_payload = base64.b64encode(b"\x00" * FRAME_SIZE).decode("ascii")
                for _ in range(SILENCE_FRAME_COUNT):
                    await websocket.send(packet("media", stream_sid=case.stream_sid, media={"payload": silence_payload, "sequence_number": sequence}))
                    sequence += 1
                    await asyncio.sleep(0.095)

                await wait_for_log(report, ("[READY] Playback complete", "[TURN COMPLETE]"), timeout=35.0)
                await asyncio.sleep(0.5)
                await websocket.send(packet("stop", stop={"reason": "automated_test_complete"}))
            finally:
                receiver_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver_task

        if len(report.response_pcm) <= SAMPLE_RATE * SAMPLE_WIDTH:
            raise AssertionError(f"Expected over one second of non-silent output for {case.language_code}")
        write_pcm_wav(case.output_path, bytes(report.response_pcm))
        if not case.output_path.is_file() or case.output_path.stat().st_size <= SAMPLE_RATE * SAMPLE_WIDTH:
            raise AssertionError(f"Invalid output WAV for {case.language_code}")
        report_case(case, report, silence_started_at)
    finally:
        active_report[0] = None


async def run_suite(port: int) -> None:
    settings = get_settings()
    if not settings.effective_bhashini_inference_api_key or not settings.groq_api_key:
        raise RuntimeError("Bhashini and GROQ_API_KEY settings must be set in .env")

    input_audio = {case.language_code: await create_input(case) for case in TEST_CASES}
    base_url = f"http://127.0.0.1:{port}"
    websocket_url = f"ws://127.0.0.1:{port}/ws/exotel-stream"
    if settings.exotel_webhook_token:
        websocket_url += f"?token={quote(settings.exotel_webhook_token)}"

    server_env = dict(os.environ)
    server_env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_DIRECTORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=server_env,
    )
    active_report: list[TestReport | None] = [None]
    log_task = asyncio.create_task(capture_server_output(process, active_report))
    try:
        await wait_for_health(base_url, process)
        suite_started = time.monotonic()
        for case in TEST_CASES:
            print(f"\n--- Running {case.language_code} ---")
            await run_case(case, websocket_url, input_audio[case.language_code], active_report)
        print(f"\nAll multilingual voicebot tests passed in {time.monotonic() - suite_started:.2f}s.")
    finally:
        if process.poll() is None:
            process.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=10.0)
            if process.poll() is None:
                process.kill()
        log_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await log_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multilingual local Exotel Voicebot integration tests")
    parser.add_argument("--port", type=int, default=8000, help="Temporary local Uvicorn port")
    arguments = parser.parse_args()
    asyncio.run(run_suite(arguments.port))


if __name__ == "__main__":
    main()
