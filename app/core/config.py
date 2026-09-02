from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Exotel Inbound Audio Service"
    environment: str = "development"
    log_level: str = "INFO"
    exotel_webhook_token: str | None = Field(default=None, repr=False)
    exotel_signature_secret: str | None = Field(default=None, repr=False)
    sarvam_api_key: str | None = Field(default=None, repr=False)
    sarvam_stt_model: str = "saaras:v3"
    sarvam_request_timeout_seconds: float = 60.0
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "priya"
    vad_rms_threshold: int = 400
    vad_silence_seconds: float = 2.0
    recording_allowed_hosts: str = ""
    recordings_directory: Path = Path("./data/recordings")
    max_recording_bytes: int = 50 * 1024 * 1024
    recording_download_timeout_seconds: float = 30.0

    @property
    def allowed_recording_hosts(self) -> set[str]:
        return {
            item.strip().lower()
            for item in self.recording_allowed_hosts.split(",")
            if item.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
