"""
OpenAI-Compatible API Models for Maya-1 TTS
Matches OpenAI's TTS API specification exactly.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class CreateSpeechRequest(BaseModel):
    """
    OpenAI-compatible speech creation request.
    
    https://platform.openai.com/docs/api-reference/audio/create-speech
    """
    model: str = Field(
        default="tts-1",
        description="The model to use for TTS. Maya-1 supports tts-1 and tts-1-hd, others will default to tts-1."
    )
    input: str = Field(
        ...,
        description="The text to generate audio for. Maximum 4096 characters.",
        min_length=1,
        max_length=4096
    )
    voice: str = Field(
        ...,
        description="The voice to use for TTS. Supported: alloy, echo, fable, onyx, nova, shimmer. Others will default to alloy."
    )
    response_format: Optional[Literal["mp3", "wav"]] = Field(
        default="mp3",
        description="The format to return the audio in"
    )
    speed: Optional[float] = Field(
        default=1.0,
        description="The speed of the generated audio. 0.25 to 4.0.",
        ge=0.25,
        le=4.0
    )


class VoiceInfo(BaseModel):
    """Information about available voices."""
    voice_id: str
    name: str
    description: str
    language: str = "en"
    gender: Optional[str] = None


class VoicesResponse(BaseModel):
    """Response for voices listing endpoint."""
    voices: list[VoiceInfo]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: dict = Field(
        ...,
        description="Error details following OpenAI format"
    )


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """Response for models listing endpoint."""
    object: str = "list"
    data: list[ModelInfo]


class UsageInfo(BaseModel):
    """Usage information for API requests."""
    prompt_tokens: int
    completion_tokens: Optional[int] = None
    total_tokens: int


# OpenAI-style error types
class OpenAIError(Exception):
    """Base OpenAI-style error."""
    def __init__(self, message: str, type: str, code: Optional[str] = None):
        self.message = message
        self.type = type
        self.code = code
        super().__init__(message)


class InvalidRequestError(OpenAIError):
    """Invalid request error."""
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message, "invalid_request_error", code)


class APIError(OpenAIError):
    """General API error."""
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message, "api_error", code)