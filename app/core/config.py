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
    bhashini_user_id: str | None = Field(default=None, repr=False)
    bhashini_ulca_api_key: str | None = Field(default=None, repr=False)
    bhashini_inference_api_key: str | None = Field(default=None, repr=False)
    # Compatibility with the key name already present in this project's .env.
    bhashini_api_key: str | None = Field(default=None, repr=False)
    bhashini_pipeline_id: str = "64392f96daac500b55c543d0"
    bhashini_stt_pipeline_id: str | None = None
    bhashini_tts_pipeline_id: str | None = None
    bhashini_ald_pipeline_id: str | None = "ai4bharat/spoken-language-identification"
    bhashini_translation_pipeline_id: str = "ai4bharat/indictrans-v2-all-gpu--t4"
    bhashini_config_url: str = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    bhashini_inference_url: str = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    bhashini_request_timeout_seconds: float = 60.0
    bhashini_tts_gender: str = "female"
    bhashini_tts_script_mode: str = "native"
    bhashini_mock_fallback: bool = True
    default_caller_language: str = "te-IN"
    server_base_url: str | None = None
    vad_rms_threshold: int = 400
    # Trigger the voice turn shortly after the caller finishes a sentence.
    vad_silence_seconds: float = 0.9
    # Groq completed the observed live RAG request in about 4.6 seconds;
    # retain a modest margin instead of discarding a valid answer prematurely.
    rag_query_timeout_seconds: float = 8.0
    # Use the attached knowledge base and LLM to answer the caller's question.
    live_rag_enabled: bool = True
    # Used only when the textbook retrieval has no sufficiently relevant context.
    # Kept separate from speech-provider credentials because it is read from .env.
    groq_api_key: str | None = Field(default=None, repr=False)
    groq_model: str = "openai/gpt-oss-20b"
    rag_min_retrieval_score: int = 2
    # 0 means "match the sample rate negotiated in Exotel's start event".
    # This prevents 8 kHz PCM from sounding garbled on a 16 kHz Voicebot stream.
    exotel_playback_sample_rate: int = 0
    recording_allowed_hosts: str = ""
    recordings_directory: Path = Path("./data/recordings")
    max_recording_bytes: int = 50 * 1024 * 1024
    recording_download_timeout_seconds: float = 30.0

    @property
    def effective_bhashini_inference_api_key(self) -> str | None:
        return self.bhashini_inference_api_key or self.bhashini_api_key

    @property
    def effective_bhashini_pipeline_id(self) -> str:
        return self.bhashini_pipeline_id or "64392f96daac500b55c543d0"

    @property
    def effective_bhashini_stt_service_id(self) -> str | None:
        return self.bhashini_stt_pipeline_id

    @property
    def effective_bhashini_tts_service_id(self) -> str | None:
        return self.bhashini_tts_pipeline_id

    def default_bhashini_tts_service_id(self, language_code: str = "en-IN") -> str:
        short = language_code.split("-", 1)[0].lower()
        if short in {"te", "ta", "kn", "ml"}:
            return "ai4bharat/indic-tts-coqui-dravidian-gpu--t4"
        return "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4"

    def default_bhashini_asr_service_id(self, language_code: str = "en-IN") -> str:
        short = language_code.split("-", 1)[0].lower()
        conformer_map = {
            "hi": "ai4bharat/conformer-hi-gpu--t4",
            "ta": "ai4bharat/conformer-ta-gpu--t4",
            "te": "ai4bharat/conformer-te-gpu--t4",
            "bn": "ai4bharat/conformer-bn-gpu--t4",
            "mr": "ai4bharat/conformer-mr-gpu--t4",
            "gu": "ai4bharat/conformer-gu-gpu--t4",
        }
        return conformer_map.get(short, "ai4bharat/whisper-medium-multilingual")

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
