"""
Voice Mapping for OpenAI to Maya-1 TTS
Maps OpenAI voice names to Maya-1 voice descriptions.
"""

from typing import Dict
from .constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS


# OpenAI voice to Maya-1 description mapping
OPENAI_VOICE_MAPPING: Dict[str, str] = {
    "alloy": (
        "Neutral male voice in their 30s with american accent. "
        "Normal pitch, warm timbre, conversational pacing, neutral tone delivery at medium intensity."
    ),
    "echo": (
        "Male voice in their 30s with american accent. "
        "Normal pitch, warm timbre, slight resonance quality, neutral tone delivery at medium intensity."
    ),
    "fable": (
        "Storyteller voice with expressive range. Male voice in their 40s with american accent. "
        "Dynamic pitch modulation, engaging timbre, theatrical pacing, expressive tone delivery."
    ),
    "onyx": (
        "Deep male voice in their 40s with american accent. "
        "Low pitch, rich timbre, slow deliberate pacing, authoritative tone delivery at high intensity."
    ),
    "nova": (
        "Female voice in their 30s with american accent. "
        "Bright pitch, clear timbre, energetic pacing, cheerful tone delivery at medium intensity."
    ),
    "shimmer": (
        "Female voice in their 30s with american accent. "
        "Soft pitch, gentle timbre, smooth pacing, warm tone delivery at low intensity."
    ),
}


# Voice-specific generation parameters
VOICE_PARAMETERS: Dict[str, Dict[str, float]] = {
    "alloy": {
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "echo": {
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "fable": {
        "temperature": DEFAULT_TEMPERATURE + 0.1,  # Slightly more creative
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "onyx": {
        "temperature": DEFAULT_TEMPERATURE - 0.1,  # Slightly more stable
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "nova": {
        "temperature": DEFAULT_TEMPERATURE + 0.05,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "shimmer": {
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
}


def get_voice_description(voice: str) -> str:
    """
    Get Maya-1 voice description for OpenAI voice name.
    
    Args:
        voice: OpenAI voice name (alloy, echo, fable, onyx, nova, shimmer)
        
    Returns:
        Maya-1 voice description string
        
    Raises:
        ValueError: If voice is not supported
    """
    if voice not in OPENAI_VOICE_MAPPING:
        raise ValueError(f"Unsupported voice: {voice}. Supported voices: {list(OPENAI_VOICE_MAPPING.keys())}")
    
    return OPENAI_VOICE_MAPPING[voice]


def get_voice_parameters(voice: str) -> Dict[str, float]:
    """
    Get generation parameters for specific voice.
    
    Args:
        voice: OpenAI voice name
        
    Returns:
        Dictionary of generation parameters
    """
    if voice not in VOICE_PARAMETERS:
        raise ValueError(f"Unsupported voice: {voice}")
    
    return VOICE_PARAMETERS[voice].copy()


def get_supported_voices() -> list[str]:
    """Get list of supported OpenAI voice names."""
    return list(OPENAI_VOICE_MAPPING.keys())