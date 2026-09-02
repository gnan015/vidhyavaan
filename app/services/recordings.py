import logging
import ipaddress
import re
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, status

from app.core.config import Settings

logger = logging.getLogger(__name__)


def validate_recording_url(url: str, settings: Settings) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise HTTPException(status_code=422, detail="RecordingUrl must be an HTTPS URL")
    with suppress(ValueError):
        address = ipaddress.ip_address(parsed.hostname)
        if not address.is_global:
            raise HTTPException(status_code=422, detail="RecordingUrl must not target a private address")
    allowed = settings.allowed_recording_hosts
    if settings.environment.lower() == "production" and not allowed:
        raise HTTPException(status_code=503, detail="Recording host allowlist is not configured")
    if allowed and parsed.hostname.lower() not in allowed:
        raise HTTPException(status_code=422, detail="RecordingUrl host is not allowlisted")


def _filename(call_sid: str, content_type: str | None) -> str:
    safe_sid = re.sub(r"[^A-Za-z0-9_.-]", "_", call_sid)
    extension = ".wav" if content_type and "wav" in content_type else ".audio"
    return f"{safe_sid}{extension}"


async def download_and_process_recording(call_sid: str, recording_url: str, settings: Settings) -> Path:
    """Download bounded audio to disk. Replace the final log with async STT processing."""
    validate_recording_url(recording_url, settings)
    destination_dir = settings.recordings_directory.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=False, timeout=settings.recording_download_timeout_seconds) as client:
        async with client.stream("GET", recording_url) as response:
            response.raise_for_status()
            declared_length = response.headers.get("content-length")
            if declared_length and int(declared_length) > settings.max_recording_bytes:
                raise ValueError("Recording exceeds configured maximum size")
            output = destination_dir / _filename(call_sid, response.headers.get("content-type"))
            total = 0
            with output.open("wb") as file:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > settings.max_recording_bytes:
                        file.close()
                        output.unlink(missing_ok=True)
                        raise ValueError("Recording exceeds configured maximum size")
                    file.write(chunk)
    logger.info("recording_downloaded", extra={"call_sid": call_sid, "recording_url": recording_url, "event": "recording_downloaded"})
    return output
