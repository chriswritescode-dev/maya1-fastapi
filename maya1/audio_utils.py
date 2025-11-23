"""
Audio Format Conversion Utilities
Handles conversion between WAV (Maya-1 native) and MP3 formats.
"""

import io
import wave
from typing import Optional, Union
from pydub import AudioSegment
from .constants import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_BITS_PER_SAMPLE


def create_wav_header(sample_rate: int = AUDIO_SAMPLE_RATE, 
                     channels: int = AUDIO_CHANNELS, 
                     bits_per_sample: int = AUDIO_BITS_PER_SAMPLE, 
                     data_size: int = 0) -> bytes:
    """
    Create WAV file header for streaming.
    
    Args:
        sample_rate: Audio sample rate (default: 24000)
        channels: Number of audio channels (default: 1)
        bits_per_sample: Bits per sample (default: 16)
        data_size: Size of audio data in bytes
        
    Returns:
        WAV header as bytes
    """
    import struct
    
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    
    return header


def wav_to_mp3(wav_data: bytes, bitrate: str = "128k") -> bytes:
    """
    Convert WAV audio data to MP3 format.
    
    Args:
        wav_data: WAV audio data as bytes
        bitrate: MP3 bitrate (64k, 96k, 128k, 192k, 320k)
        
    Returns:
        MP3 audio data as bytes
        
    Raises:
        Exception: If conversion fails
    """
    try:
        # Load WAV data into AudioSegment
        wav_audio = AudioSegment.from_wav(io.BytesIO(wav_data))
        
        # Export to MP3
        mp3_buffer = io.BytesIO()
        wav_audio.export(mp3_buffer, format="mp3", bitrate=bitrate)
        
        return mp3_buffer.getvalue()
        
    except Exception as e:
        # Check if ffmpeg is missing
        if "No such file or directory" in str(e) and "ffmpeg" in str(e).lower():
            raise Exception("MP3 conversion not available: ffmpeg not installed. Please install ffmpeg or request WAV format.")
        raise Exception(f"Failed to convert WAV to MP3: {str(e)}")


def mp3_to_wav(mp3_data: bytes) -> bytes:
    """
    Convert MP3 audio data to WAV format.
    
    Args:
        mp3_data: MP3 audio data as bytes
        
    Returns:
        WAV audio data as bytes
        
    Raises:
        Exception: If conversion fails
    """
    try:
        # Load MP3 data into AudioSegment
        mp3_audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        
        # Export to WAV
        wav_buffer = io.BytesIO()
        mp3_audio.export(wav_buffer, format="wav", 
                        parameters=["-ar", str(AUDIO_SAMPLE_RATE), 
                                   "-ac", str(AUDIO_CHANNELS)])
        
        return wav_buffer.getvalue()
        
    except Exception as e:
        raise Exception(f"Failed to convert MP3 to WAV: {str(e)}")


def create_wav_bytes(audio_data: bytes) -> bytes:
    """
    Create proper WAV file from raw audio data.
    
    Args:
        audio_data: Raw audio data (int16 PCM)
        
    Returns:
        Complete WAV file as bytes
    """
    wav_buffer = io.BytesIO()
    
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(AUDIO_CHANNELS)
        wav_file.setsampwidth(AUDIO_BITS_PER_SAMPLE // 8)
        wav_file.setframerate(AUDIO_SAMPLE_RATE)
        wav_file.writeframes(audio_data)
    
    wav_buffer.seek(0)
    return wav_buffer.getvalue()


def convert_audio_format(audio_data: bytes, 
                        input_format: str, 
                        output_format: str,
                        bitrate: str = "128k") -> bytes:
    """
    Convert audio between formats.
    
    Args:
        audio_data: Input audio data as bytes
        input_format: Input format ('wav' or 'mp3')
        output_format: Output format ('wav' or 'mp3')
        bitrate: MP3 bitrate (only used for MP3 output)
        
    Returns:
        Converted audio data as bytes
        
    Raises:
        ValueError: If formats are not supported
        Exception: If conversion fails
    """
    if input_format == output_format:
        return audio_data
    
    if input_format == "wav" and output_format == "mp3":
        return wav_to_mp3(audio_data, bitrate)
    elif input_format == "mp3" and output_format == "wav":
        return mp3_to_wav(audio_data)
    else:
        raise ValueError(f"Unsupported format conversion: {input_format} -> {output_format}")


def get_audio_mime_type(format: str) -> str:
    """
    Get MIME type for audio format.
    
    Args:
        format: Audio format ('wav' or 'mp3')
        
    Returns:
        MIME type string
        
    Raises:
        ValueError: If format is not supported
    """
    mime_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg"
    }
    
    if format not in mime_types:
        raise ValueError(f"Unsupported audio format: {format}")
    
    return mime_types[format]


def get_audio_extension(format: str) -> str:
    """
    Get file extension for audio format.
    
    Args:
        format: Audio format ('wav' or 'mp3')
        
    Returns:
        File extension string
        
    Raises:
        ValueError: If format is not supported
    """
    extensions = {
        "wav": ".wav",
        "mp3": ".mp3"
    }
    
    if format not in extensions:
        raise ValueError(f"Unsupported audio format: {format}")
    
    return extensions[format]