"""
Maya1 Streaming Pipeline - Sliding Window Approach
Implements sliding window technique for smooth streaming without artifacts.
"""

import asyncio
from typing import AsyncGenerator, Optional
from vllm import SamplingParams

from .constants import (
    CODE_END_TOKEN_ID,
    SNAC_MIN_ID,
    SNAC_MAX_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
)


class Maya1SlidingWindowPipeline:
    """
    Streaming TTS pipeline using sliding window approach.
    Decodes overlapping 28-token windows (4 frames) and keeps only 
    the middle 2048 samples for smooth audio continuity.
    """
    
    # Sliding window configuration
    WINDOW_SIZE = 28  # 4 frames (7 tokens per frame)
    YIELD_STRIDE = 7  # Yield every 1 frame
    MIDDLE_SAMPLES = 2048  # Keep middle 2048 samples from each decode
    
    def __init__(self, model, prompt_builder, snac_decoder):
        """
        Initialize sliding window streaming pipeline.
        
        Args:
            model: Maya1Model instance
            prompt_builder: Maya1PromptBuilder instance
            snac_decoder: SNACDecoder instance
        """
        self.model = model
        self.prompt_builder = prompt_builder
        self.snac_decoder = snac_decoder
        print(f"Sliding window pipeline initialized")
    
    async def generate_speech_stream(
        self,
        description: str,
        text: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: Optional[int] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech audio with sliding window streaming.
        
        Args:
            description: Voice description
            text: Text to synthesize (may include <emotion> tags)
            temperature: Sampling temperature
            top_p: Nucleus sampling
            max_tokens: Max SNAC tokens to generate
            repetition_penalty: Prevent loops
            seed: Random seed
        
        Yields:
            Audio bytes (int16 PCM, 24kHz mono)
        """
        # Build prompt
        prompt = self.prompt_builder.build_prefix(description, text)
        
        # Configure sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=DEFAULT_MIN_TOKENS,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[CODE_END_TOKEN_ID],
            seed=seed if seed is not None else DEFAULT_SEED,
        )
        
        # Stream tokens
        snac_buffer = []
        last_yield_position = 0
        chunk_count = 0
        total_tokens_seen = 0
        
        async for output in self.model.generate_stream(prompt, sampling_params):
            # Get latest generated tokens (cumulative list)
            generated_token_ids = output.outputs[0].token_ids
            
            # Process only NEW tokens since last iteration
            new_tokens = generated_token_ids[total_tokens_seen:]
            total_tokens_seen = len(generated_token_ids)
            
            # Collect SNAC codes from new tokens
            for token_id in new_tokens:
                # Stop if we hit EOS
                if token_id == CODE_END_TOKEN_ID:
                    break
                
                # Only collect valid SNAC tokens
                if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID:
                    snac_buffer.append(token_id)
            
            # Yield audio when we have enough tokens for a window
            while len(snac_buffer) >= last_yield_position + self.WINDOW_SIZE:
                # Get window of 28 tokens
                window_start = last_yield_position
                window_end = window_start + self.WINDOW_SIZE
                window = snac_buffer[window_start:window_end]
                
                if len(window) == self.WINDOW_SIZE:
                    # Decode window to audio
                    audio_bytes = await self.snac_decoder.decode_single_async(window)
                    
                    if audio_bytes:
                        # Extract middle portion of audio
                        audio_samples = len(audio_bytes) // 2
                        middle_start_sample = (audio_samples - self.MIDDLE_SAMPLES) // 2
                        middle_end_sample = middle_start_sample + self.MIDDLE_SAMPLES
                        
                        # Convert to byte positions
                        middle_start_byte = middle_start_sample * 2
                        middle_end_byte = middle_end_sample * 2
                        
                        # Extract middle chunk
                        audio_chunk = audio_bytes[middle_start_byte:middle_end_byte]
                        
                        chunk_count += 1
                        if chunk_count == 1:
                            print(f" First chunk ready")
                        
                        yield audio_chunk
                
                # Move forward by stride
                last_yield_position += self.YIELD_STRIDE
            
            # Check if generation is done
            if CODE_END_TOKEN_ID in new_tokens:
                break
        
        # Final chunk: decode remaining tokens
        remaining_tokens = len(snac_buffer) - last_yield_position
        if remaining_tokens >= self.WINDOW_SIZE:
            window = snac_buffer[-self.WINDOW_SIZE:]
            audio_bytes = await self.snac_decoder.decode_single_async(window)
            if audio_bytes:
                yield audio_bytes[-self.MIDDLE_SAMPLES * 2:]
        
        frames = len(snac_buffer) // 7
        duration = frames / 6.86
        print(f"Streamed {chunk_count} chunks (~{duration:.1f}s audio)")