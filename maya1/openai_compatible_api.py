"""
OpenAI-Compatible Maya-1 TTS API
Replaces the original Maya-1 API with OpenAI-compatible endpoints.
"""

import os
import io
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .model_loader import Maya1Model
from .prompt_builder import Maya1PromptBuilder
from .snac_decoder import SNACDecoder
from .pipeline import Maya1Pipeline
from .streaming_pipeline import Maya1SlidingWindowPipeline
from .openai_api import (
    CreateSpeechRequest, 
    VoicesResponse, 
    VoiceInfo, 
    ModelsResponse, 
    ModelInfo,
    ErrorResponse,
    InvalidRequestError,
    APIError
)
from .voice_mapping import get_voice_description, get_voice_parameters, get_supported_voices
from .audio_utils import (
    create_wav_header, 
    convert_audio_format, 
    get_audio_mime_type, 
    get_audio_extension,
    create_wav_bytes
)
from .constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    AUDIO_SAMPLE_RATE,
)

# Timeout settings (seconds)
GENERATE_TIMEOUT = 60

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Maya-1 TTS API",
    description="OpenAI-compatible TTS API for Maya-1 Voice",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
    
    print("\\n" + "="*60)
    print(" Starting Maya-1 OpenAI-Compatible TTS API Server")
    print("="*60 + "\\n")
    
    # Initialize components
    model = Maya1Model()
    prompt_builder = Maya1PromptBuilder(model.tokenizer, model)
    
    # Initialize SNAC decoder
    snac_decoder = SNACDecoder(enable_batching=True, max_batch_size=64, batch_timeout_ms=15)
    await snac_decoder.start_batch_processor()
    
    # Initialize pipelines
    pipeline = Maya1Pipeline(model, prompt_builder, snac_decoder)
    streaming_pipeline = Maya1SlidingWindowPipeline(model, prompt_builder, snac_decoder)
    
    print("\\n" + "="*60)
    print("Maya-1 OpenAI-Compatible TTS API Server Ready")
    print("="*60 + "\\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\nShutting down Maya-1 TTS API Server")
    
    # Kill VLLM process if we have the PID
    if model and hasattr(model, 'vllm_pid') and model.vllm_pid:
        try:
            import os
            import signal
            print(f"Terminating VLLM process (PID: {model.vllm_pid})")
            os.kill(model.vllm_pid, signal.SIGTERM)
            # Give it a moment to shut down gracefully
            import asyncio
            await asyncio.sleep(2)
            # Force kill if still running
            try:
                os.kill(model.vllm_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Already terminated
        except (ProcessLookupError, OSError):
            pass  # Process already dead
    
    if snac_decoder and snac_decoder.is_running:
        await snac_decoder.stop_batch_processor()


# ============================================================================
# OpenAI-Compatible Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with OpenAI-style response."""
    return {
        "object": "api",
        "version": "1.0.0",
        "owner": "maya-1",
        "models": [
            {"id": "tts-1", "object": "model"},
            {"id": "tts-1-hd", "object": "model"}
        ]
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id="tts-1",
                created=int(time.time()),
                owned_by="maya-1"
            ),
            ModelInfo(
                id="tts-1-hd", 
                created=int(time.time()),
                owned_by="maya-1"
            )
        ]
    )


@app.get("/v1/voices")
async def list_voices():
    """List available voices (OpenAI-compatible extension)."""
    voices = []
    for voice_name in get_supported_voices():
        description = get_voice_description(voice_name)
        voices.append(VoiceInfo(
            voice_id=voice_name,
            name=voice_name.capitalize(),
            description=description.split('.')[0] + '.',  # First sentence only
            language="en"
        ))
    
    return VoicesResponse(voices=voices)


@app.post("/v1/audio/speech")
async def create_speech(request: CreateSpeechRequest):
    """
    Create speech from text (OpenAI-compatible endpoint).
    
    This endpoint matches OpenAI's TTS API specification:
    https://platform.openai.com/docs/api-reference/audio/create-speech
    """
    try:
        # Validate input
        if len(request.input) > 4096:
            raise InvalidRequestError("Input text exceeds maximum length of 4096 characters", "max_length_exceeded")
        
        if request.speed and (request.speed < 0.25 or request.speed > 4.0):
            raise InvalidRequestError("Speed must be between 0.25 and 4.0", "invalid_speed")
        
        # Handle invalid model names gracefully - default to tts-1
        model = request.model
        if model not in ["tts-1", "tts-1-hd"]:
            print(f"⚠️  Unknown model '{model}', defaulting to 'tts-1'")
            model = "tts-1"
        
        # Handle invalid voice names gracefully - default to alloy
        voice = request.voice
        try:
            voice_description = get_voice_description(voice)
            voice_params = get_voice_parameters(voice)
        except (KeyError, ValueError):
            print(f"⚠️  Unknown voice '{voice}', defaulting to 'alloy'")
            voice = "alloy"
            voice_description = get_voice_description(voice)
            voice_params = get_voice_parameters(voice)
        
        # Adjust parameters based on speed
        # Speed affects how we generate the audio
        adjusted_max_tokens = int(DEFAULT_MAX_TOKENS / request.speed) if request.speed and request.speed > 0 else DEFAULT_MAX_TOKENS
        
        print(f"🎙️  Generating speech: voice={voice}, model={model}, format={request.response_format}")
        print(f"📝 Input: {request.input[:100]}{'...' if len(request.input) > 100 else ''}")
        
        # Generate audio using Maya-1 pipeline
        if model == "tts-1-hd":
            # Use higher quality settings for HD model
            audio_bytes = await _generate_complete_audio(
                description=voice_description,
                text=request.input,
                temperature=voice_params["temperature"] - 0.05,  # Slightly more stable
                top_p=voice_params["top_p"],
                max_tokens=adjusted_max_tokens,
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
            )
        else:
            # Standard tts-1 model
            audio_bytes = await _generate_complete_audio(
                description=voice_description,
                text=request.input,
                temperature=voice_params["temperature"],
                top_p=voice_params["top_p"],
                max_tokens=adjusted_max_tokens,
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
            )
        
        if audio_bytes is None:
            raise APIError("Audio generation failed")
        
        # Convert to requested format
        if request.response_format == "mp3":
            try:
                audio_bytes = convert_audio_format(audio_bytes, "wav", "mp3")
            except Exception as e:
                if "ffmpeg not installed" in str(e):
                    # Fallback to WAV if ffmpeg is not available
                    print("⚠️  MP3 conversion not available, falling back to WAV format")
                    request.response_format = "wav"
                else:
                    raise
        
        # Return streaming response
        response_format = request.response_format or "wav"
        mime_type = get_audio_mime_type(response_format)
        filename = f"speech{get_audio_extension(response_format)}"
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=mime_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except (InvalidRequestError, APIError):
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        raise APIError(f"Internal server error: {str(e)}")


async def _generate_complete_audio(
    description: str,
    text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> Optional[bytes]:
    """Generate complete audio using Maya-1 pipeline."""
    try:
        import asyncio
        
        # Generate audio
        if not pipeline:
            raise APIError("Pipeline not initialized")
            
        audio_bytes = await asyncio.wait_for(
            pipeline.generate_speech(
                description=description,
                text=text,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                seed=None,
            ),
            timeout=GENERATE_TIMEOUT
        )
        
        if audio_bytes is None:
            return None
        
        # Create proper WAV file
        return create_wav_bytes(audio_bytes)
    
    except Exception as e:
        if "TimeoutError" in str(type(e)):
            raise APIError("Generation timeout", "timeout")
        raise


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    vllm_pid = None
    if model and hasattr(model, 'vllm_pid'):
        vllm_pid = model.vllm_pid
    
    return {
        "status": "healthy",
        "model": "Maya-1-Voice",
        "openai_compatible": True,
        "vllm_pid": vllm_pid,
        "timestamp": time.time(),
    }


@app.get("/v1/pid")
async def get_vllm_pid():
    """Get VLLM process PID for process management."""
    if not model or not hasattr(model, 'vllm_pid') or not model.vllm_pid:
        raise HTTPException(status_code=404, detail="VLLM PID not available")
    
    return {"vllm_pid": model.vllm_pid}


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(InvalidRequestError)
async def invalid_request_handler(request, exc: InvalidRequestError):
    """Handle invalid request errors."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error={
                "type": exc.type,
                "message": exc.message,
                "code": exc.code
            }
        ).dict()
    )


@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    """Handle API errors."""
    return HTTPException(
        status_code=500,
        detail=ErrorResponse(
            error={
                "type": exc.type,
                "message": exc.message,
                "code": exc.code
            }
        ).dict()
    )


# ============================================================================
# Direct Run Support
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )