import os
import io
import wave
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .model_loader import Maya1Model
from .prompt_builder import Maya1PromptBuilder
from .snac_decoder import SNACDecoder
from .pipeline import Maya1Pipeline
from .streaming_pipeline import Maya1SlidingWindowPipeline
from .constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    AUDIO_SAMPLE_RATE,
    OPENAI_VOICE_MAPPINGS,
    OPENAI_SUPPORTED_VOICES,
    OPENAI_SUPPORTED_FORMATS,
    OPENAI_DEFAULT_MODEL,
    OPENAI_SUPPORTED_MODELS,
)

# Timeout settings (seconds)
GENERATE_TIMEOUT = 60

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Maya1 TTS API",
    description="Open source TTS inference for Maya1",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
prompt_builder = None
snac_decoder = None
pipeline = None
streaming_pipeline = None


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, prompt_builder, snac_decoder, pipeline, streaming_pipeline
    
    print("\n" + "="*60)
    print(" Starting Maya1 TTS API Server")
    print("="*60 + "\n")
    
    # Initialize components
    model = Maya1Model()
    prompt_builder = Maya1PromptBuilder(model.tokenizer, model)
    
    # Initialize SNAC decoder
    snac_decoder = SNACDecoder(enable_batching=True, max_batch_size=64, batch_timeout_ms=15)
    await snac_decoder.start_batch_processor()
    
    # Initialize pipelines
    pipeline = Maya1Pipeline(model, prompt_builder, snac_decoder)
    streaming_pipeline = Maya1SlidingWindowPipeline(model, prompt_builder, snac_decoder)
    
    print("\n" + "="*60)
    print("Maya1 TTS API Server Ready")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\nShutting down Maya1 TTS API Server")
    
    if snac_decoder and snac_decoder.is_running:
        await snac_decoder.stop_batch_processor()


# ============================================================================
# Utility Functions
# ============================================================================

def create_wav_header(sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16, data_size: int = 0) -> bytes:
    """Create WAV file header."""
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
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    
    return header


def convert_audio_format(audio_bytes: bytes, input_format: str = "wav", output_format: str = "mp3", sample_rate: int = 24000) -> bytes:
    """Convert audio between different formats using pydub."""
    try:
        from pydub import AudioSegment
        import io
        
        # Create AudioSegment from raw audio bytes
        if input_format == "wav":
            audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        elif input_format == "mp3":
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        elif input_format == "flac":
            audio = AudioSegment.from_flac(io.BytesIO(audio_bytes))
        else:
            # For raw PCM or other formats, create from raw data
            audio = AudioSegment(
                audio_bytes,
                sample_width=2,  # 16-bit
                frame_rate=sample_rate,
                channels=1
            )
        
        # Export to desired format
        output_buffer = io.BytesIO()
        if output_format == "mp3":
            audio.export(output_buffer, format="mp3", bitrate="128k")
        elif output_format == "opus":
            audio.export(output_buffer, format="opus")
        elif output_format == "aac":
            audio.export(output_buffer, format="aac")
        elif output_format == "flac":
            audio.export(output_buffer, format="flac")
        elif output_format == "wav":
            audio.export(output_buffer, format="wav")
        elif output_format == "pcm":
            # Return raw PCM data
            return audio.raw_data
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return output_buffer.getvalue()
    
    except ImportError:
        # If pydub is not available, return original audio (WAV format)
        if output_format == "wav":
            return audio_bytes
        else:
            raise HTTPException(
                status_code=501, 
                detail=f"Audio format conversion to {output_format} requires pydub. Install with: pip install pydub"
            )


def adjust_audio_speed(audio_bytes: bytes, speed: float, sample_rate: int = 24000) -> bytes:
    """Adjust audio playback speed."""
    if speed == 1.0:
        return audio_bytes
    
    try:
        from pydub import AudioSegment
        import io
        
        # Load audio
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        
        # Adjust speed by changing frame rate
        if speed > 1.0:
            # Faster: increase frame rate
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * speed)
            })
        else:
            # Slower: decrease frame rate
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * speed)
            })
        
        # Convert back to original sample rate
        audio = audio.set_frame_rate(sample_rate)
        
        # Export back to WAV
        output_buffer = io.BytesIO()
        audio.export(output_buffer, format="wav")
        return output_buffer.getvalue()
    
    except ImportError:
        # If pydub is not available, return original audio
        return audio_bytes


def get_content_type_for_format(format: str) -> str:
    """Get appropriate MIME type for audio format."""
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm"
    }
    return content_types.get(format, "application/octet-stream")


# ============================================================================
# Request/Response Models
# ============================================================================

class TTSRequest(BaseModel):
    """TTS generation request."""
    description: str = Field(
        ...,
        description="Voice description (e.g., 'Male voice in their 30s with american accent')"
    )
    text: str = Field(
        ...,
        description="Text to synthesize (can include <emotion> tags)"
    )
    temperature: Optional[float] = Field(
        default=DEFAULT_TEMPERATURE,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=DEFAULT_TOP_P,
        description="Nucleus sampling"
    )
    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Maximum tokens to generate"
    )
    repetition_penalty: Optional[float] = Field(
        default=DEFAULT_REPETITION_PENALTY,
        description="Repetition penalty"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
        ge=0,
    )
    stream: bool = Field(
        default=False,
        description="Stream audio (True) or return complete WAV (False)"
    )


class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS request."""
    model: str = Field(
        default=OPENAI_DEFAULT_MODEL,
        description=f"Model to use for TTS. Supported: {', '.join(OPENAI_SUPPORTED_MODELS)}"
    )
    input: str = Field(
        ...,
        description="The text to generate audio for. Maximum length: 4096 characters."
    )
    voice: str = Field(
        ...,
        description=f"The voice to use for generation. Supported: {', '.join(OPENAI_SUPPORTED_VOICES)}"
    )
    response_format: str = Field(
        default="mp3",
        description=f"The format to audio in. Supported: {', '.join(OPENAI_SUPPORTED_FORMATS)}"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. 1.0 is normal speed."
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Maya1 TTS API",
        "version": "1.0.0",
        "status": "running",
        "model": "Maya1-Voice (open source)",
        "endpoints": {
            "maya1_generate": "/v1/tts/generate (POST) - Maya1 native API",
            "openai_speech": "/v1/audio/speech (POST) - OpenAI compatible API",
            "health": "/health (GET)",
        },
        "openai_compatibility": {
            "voices": OPENAI_SUPPORTED_VOICES,
            "formats": OPENAI_SUPPORTED_FORMATS,
            "models": OPENAI_SUPPORTED_MODELS,
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "Maya1-Voice",
        "timestamp": time.time(),
    }


# ============================================================================
# TTS Generation Endpoints
# ============================================================================

@app.post("/v1/tts/generate")
async def generate_tts(request: TTSRequest):
    """Generate TTS audio from description and text (Maya1 native API)."""
    
    try:
        # Route to streaming or non-streaming
        if request.stream:
            return await _generate_tts_streaming(
                description=request.description,
                text=request.text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                seed=request.seed,
            )
        else:
            return await _generate_tts_complete(
                description=request.description,
                text=request.text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                seed=request.seed,
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f" Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech")
async def openai_tts(request: OpenAITTSRequest):
    """OpenAI-compatible TTS endpoint."""
    
    try:
        # Validate inputs
        if request.model not in OPENAI_SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Supported models: {', '.join(OPENAI_SUPPORTED_MODELS)}"
            )
        
        if request.voice not in OPENAI_SUPPORTED_VOICES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported voice: {request.voice}. Supported voices: {', '.join(OPENAI_SUPPORTED_VOICES)}"
            )
        
        if request.response_format not in OPENAI_SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {request.response_format}. Supported formats: {', '.join(OPENAI_SUPPORTED_FORMATS)}"
            )
        
        if len(request.input) > 4096:
            raise HTTPException(
                status_code=400,
                detail="Input text exceeds maximum length of 4096 characters"
            )
        
        # Map OpenAI voice to Maya1 description
        description = OPENAI_VOICE_MAPPINGS[request.voice]
        
        # Generate audio using Maya1 pipeline
        audio_bytes = await pipeline.generate_speech(
            description=description,
            text=request.input,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_TOKENS,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            seed=None,
        )
        
        if audio_bytes is None:
            raise HTTPException(status_code=500, detail="Audio generation failed")
        
        # Adjust speed if needed
        if request.speed != 1.0:
            audio_bytes = adjust_audio_speed(audio_bytes, request.speed)
        
        # Convert to requested format
        if request.response_format != "wav":
            audio_bytes = convert_audio_format(audio_bytes, "wav", request.response_format)
        
        # Return appropriate response
        content_type = get_content_type_for_format(request.response_format)
        
        if request.response_format == "pcm":
            return StreamingResponse(
                io.BytesIO(audio_bytes),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=output.{request.response_format}"
                }
            )
        else:
            return StreamingResponse(
                io.BytesIO(audio_bytes),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f" OpenAI TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_tts_complete(
    description: str,
    text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    seed: Optional[int],
):
    """Generate complete WAV file (non-streaming)."""
    
    try:
        import asyncio
        
        # Generate audio
        audio_bytes = await asyncio.wait_for(
            pipeline.generate_speech(
                description=description,
                text=text,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
            timeout=GENERATE_TIMEOUT
        )
        
        if audio_bytes is None:
            raise Exception("Audio generation failed")
        
        # Create WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(AUDIO_SAMPLE_RATE)
            wav_file.writeframes(audio_bytes)
        
        wav_buffer.seek(0)
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timeout")


async def _generate_tts_streaming(
    description: str,
    text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    seed: Optional[int],
):
    """Generate streaming audio."""
    start_time = time.time()
    first_audio_time = None
    
    async def audio_stream_generator():
        """Generate audio stream with WAV header."""
        nonlocal first_audio_time
        
        # Send WAV header first
        yield create_wav_header(sample_rate=AUDIO_SAMPLE_RATE, channels=1, bits_per_sample=16)
        
        # Stream audio chunks
        async for audio_chunk in streaming_pipeline.generate_speech_stream(
            description=description,
            text=text,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ):
            if first_audio_time is None:
                first_audio_time = time.time()
                ttfb_ms = (first_audio_time - start_time) * 1000
                print(f"⏱️  TTFB: {ttfb_ms:.1f}ms")
            
            yield audio_chunk
    
    try:
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"}
        )
    
    except Exception as e:
        print(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )