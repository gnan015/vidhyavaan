from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, field_validator

CallSid = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
PhoneNumber = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class ExotelCallback(BaseModel):
    """Normalized subset of common Passthru and Record Applet fields."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    call_sid: CallSid = Field(validation_alias="CallSid")
    from_number: PhoneNumber | None = Field(default=None, validation_alias="From")
    to_number: PhoneNumber | None = Field(default=None, validation_alias="To")
    recording_url: HttpUrl | None = Field(default=None, validation_alias="RecordingUrl")
    digits: str | None = Field(default=None, validation_alias="Digits", max_length=128)
    duration: int | None = Field(default=None, validation_alias="Duration", ge=0, le=86_400)

    @field_validator("digits", mode="before")
    @classmethod
    def normalize_digits(cls, value: object) -> str | None:
        if value is None:
            return None
        return str(value).strip() or None


class AudioFrameInfo(BaseModel):
    encoding: str
    sample_rate_hz: int
    channels: int = 1
    byte_count: int

