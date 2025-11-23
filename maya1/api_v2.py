"""
OpenAI-Compatible Maya-1 TTS API
Replaces the original Maya-1 API with OpenAI-compatible endpoints.
"""

import io
import time
import torch
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
def split_text_into_chunks(text: str, max_chunk_size: int = 1800):
    """Split text into chunks while preserving sentence boundaries."""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences
    import re
    sentences = re.split(r'([.!?]+)\s*', text)
    
    # Reconstruct sentences with punctuation
    reconstructed_sentences = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        if sentence.strip():
            reconstructed_sentences.append(sentence.strip())
    
    for sentence in reconstructed_sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Force split long sentence
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > max_chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            while len(word) > max_chunk_size:
                                chunks.append(word[:max_chunk_size])
                                word = word[max_chunk_size:]
                            temp_chunk = word
                    else:
                        temp_chunk += (" " + word) if temp_chunk else word
                current_chunk = temp_chunk
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
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
    convert_audio_format, 
    get_audio_mime_type, 
    get_audio_extension,
    create_wav_bytes
)
from .constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPETITION_PENALTY,
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
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
model = None
prompt_builder = None
snac_decoder = None
pipeline = None
streaming_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize Maya1 model and components."""
    global model, prompt_builder, snac_decoder, pipeline, streaming_pipeline
    
    print("🚀 Initializing Maya1 TTS API Server")
    
    # Initialize Maya1 model
    model = Maya1Model()
    
    # Initialize prompt builder
    prompt_builder = Maya1PromptBuilder()
    
    # Initialize SNAC decoder with batching
    snac_decoder = SNACDecoder(
        enable_batching=True, 
        max_batch_size=16, 
        batch_timeout_ms=15
    )
    
    # Initialize pipelines
    pipeline = Maya1Pipeline(model, prompt_builder, snac_decoder)
    streaming_pipeline = Maya1SlidingWindowPipeline(model, prompt_builder, snac_decoder)
    
    print("✅ Maya1 TTS API Server ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\\nShutting down Maya1 TTS API Server")
    
    if snac_decoder and snac_decoder.is_running:
        await snac_decoder.stop_batch_processor()
    
    # Clean up VLLM engine
    global model
    if model and hasattr(model, 'engine'):
        try:
            # Cancel all pending requests
            if hasattr(model.engine, 'abort_request'):
                # Get all active request IDs and abort them
                pass  # vLLM will handle cleanup on process exit
        except Exception as e:
            print(f"Error during VLLM cleanup: {e}")

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
            {"id": "tts-1-hd", "object": "model"},
            {"id": "maya-1", "object": "model"}
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
            ),
            ModelInfo(
                id="maya-1",
                created=int(time.time()),
                owned_by="maya-1"
            )
        ]
    )


@app.get("/v1/voices")
async def list_voices():
    """List available voices (OpenAI-compatible)."""
    voices = []
    for voice_id in get_supported_voices():
        voices.append(VoiceInfo(
            voice=voice_id,
            created=int(time.time())
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
        # Validate speed
        if request.speed and (request.speed < 0.25 or request.speed > 4.0):
            raise InvalidRequestError("Speed must be between 0.25 and 4.0", "invalid_speed")
        
        # Get voice description and parameters
        voice_description = get_voice_description(request.voice)
        voice_params = get_voice_parameters(request.voice)
        
        # Auto-chunk large inputs instead of rejecting them
        if len(request.input) > 2000:
            print(f"📝 Long input detected ({len(request.input)} chars), auto-chunking...")
            audio_bytes = await _generate_chunked_audio(
                description=voice_description,
                text=request.input,
                voice_params=voice_params,
                model=request.model,
                speed=float(request.speed) if request.speed else 1.0
            )
        else:
            # Adjust parameters based on speed
            speed = float(request.speed) if request.speed else 1.0
            adjusted_max_tokens = int(DEFAULT_MAX_TOKENS / speed)
            
            print(f"🎙️  Generating speech: voice={request.voice}, model={request.model}, format={request.response_format}")
            print(f"📝 Input: {request.input[:100]}{'...' if len(request.input) > 100 else ''}")
            
            # Generate audio using Maya-1 pipeline
            if request.model == "tts-1-hd":
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
            audio_bytes = convert_audio_format(audio_bytes, "wav", "mp3")
        
        # Return streaming response
        mime_type = get_audio_mime_type(request.response_format)
        filename = f"speech{get_audio_extension(request.response_format)}"
        
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


async def _generate_chunked_audio(
    description: str,
    text: str,
    voice_params: dict,
    model: str,
    speed: float,
) -> Optional[bytes]:
    """
    Generate audio by automatically chunking large text inputs.
    
    Args:
        description: Voice description
        text: Text to synthesize (may be very long)
        voice_params: Voice parameters
        model: Model name
        speed: Speed multiplier
    
    Returns:
        Combined audio bytes or None if failed
    """
    try:
        import asyncio
        
        # Split text into manageable chunks
        chunks = split_text_into_chunks(text)
        print(f"📄 Split into {len(chunks)} chunks")
        
        # Generate audio for each chunk
        audio_chunks = []
        adjusted_max_tokens = int(DEFAULT_MAX_TOKENS / speed)
        
        for i, chunk in enumerate(chunks):
            print(f"🎵 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            # Generate audio for this chunk
            chunk_audio = await asyncio.wait_for(
                pipeline.generate_speech(
                    description=description,
                    text=chunk,
                    temperature=voice_params["temperature"] - 0.05 if model == "tts-1-hd" else voice_params["temperature"],
                    top_p=voice_params["top_p"],
                    max_tokens=adjusted_max_tokens,
                    repetition_penalty=DEFAULT_REPETITION_PENALTY,
                    seed=None,
                ),
                timeout=GENERATE_TIMEOUT
            )
            
            if chunk_audio is None:
                print(f"❌ Failed to generate audio for chunk {i+1}")
                continue
            
            audio_chunks.append(chunk_audio)
        
        if not audio_chunks:
            print("❌ No audio chunks were generated successfully")
            return None
        
        # Combine all audio chunks
        print(f"🔗 Combining {len(audio_chunks)} audio chunks")
        combined_audio = b''.join(audio_chunks)
        
        # Clean up memory
        del audio_chunks
        torch.cuda.empty_cache()
        
        # Create proper WAV file
        return create_wav_bytes(combined_audio)
        
    except asyncio.TimeoutError:
        raise APIError("Generation timeout", "timeout")
    except Exception as e:
        print(f"Error in chunked audio generation: {e}")
        raise APIError(f"Chunked generation failed: {str(e)}")


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
        wav_bytes = create_wav_bytes(audio_bytes)
        
        # Clean up GPU memory
        del audio_bytes
        torch.cuda.empty_cache()
        
        return wav_bytes
    
    except asyncio.TimeoutError:
        raise APIError("Generation timeout", "timeout")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "Maya-1-Voice",
        "openai_compatible": True,
        "timestamp": time.time()
    }


# ============================================================================
# Original Maya1 Endpoints (for backward compatibility)
# ============================================================================

class TTSRequest(BaseModel):
    """Original Maya1 TTS request format."""
    model: str = Field(default="maya1", description="Model name")
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="alloy", description="Voice name")
    temperature: Optional[float] = Field(default=0.4, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Nucleus sampling")
    max_tokens: Optional[int] = Field(default=2048, description="Max tokens to generate")
    repetition_penalty: Optional[float] = Field(default=1.1, description="Repetition penalty")


@app.post("/v1/tts/generate")
async def generate_speech_legacy(request: TTSRequest):
    """Legacy Maya1 TTS endpoint for backward compatibility."""
    try:
        # Convert to OpenAI format and use new endpoint
        openai_request = CreateSpeechRequest(
            model="tts-1" if request.model == "maya1" else request.model,
            input=request.text,
            voice=request.voice
        )
        
        # Set speed based on temperature (rough approximation)
        if request.temperature:
            openai_request.speed = max(0.25, min(4.0, 1.0 / (request.temperature + 0.1)))
        
        return await create_speech(openai_request)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)