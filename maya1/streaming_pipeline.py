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
        
        snac_buffer = []
        last_yield_position = 0
        chunk_count = 0
        total_tokens_seen = 0
        cleanup_counter = 0
        end_token_seen = False
        
        async for output in self.model.generate_stream(prompt, sampling_params):
            generated_token_ids = output.outputs[0].token_ids
            
            new_tokens = generated_token_ids[total_tokens_seen:]
            total_tokens_seen = len(generated_token_ids)
            
            for token_id in new_tokens:
                if token_id == CODE_END_TOKEN_ID:
                    end_token_seen = True
                    break
                
                if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID:
                    snac_buffer.append(token_id)
            
            while len(snac_buffer) >= last_yield_position + self.WINDOW_SIZE:
                window_start = last_yield_position
                window_end = window_start + self.WINDOW_SIZE
                window = snac_buffer[window_start:window_end]
                
                if len(window) == self.WINDOW_SIZE:
                    # Use sliding window mode for proper batching
                    audio_bytes = await self.snac_decoder.decode_single_async(
                        window, 
                        trim_warmup=False,  # Don't trim, we handle it here
                        use_sliding_window=True  # Tell decoder this is sliding window
                    )
                    
                    if audio_bytes:
                        # When use_sliding_window=True, decoder already returns middle 2048 samples
                        # So we can directly use the audio_bytes
                        chunk_count += 1
                        if chunk_count == 1:
                            print(f" First chunk ready")
                        
                        yield audio_bytes
                        
                        del audio_bytes
                
                last_yield_position += self.YIELD_STRIDE
                
                if last_yield_position > self.WINDOW_SIZE * 2:
                    tokens_to_keep = last_yield_position - self.WINDOW_SIZE
                    snac_buffer = snac_buffer[tokens_to_keep:]
                    last_yield_position = self.WINDOW_SIZE
                
                cleanup_counter += 1
                if cleanup_counter % 10 == 0:
                    import gc
                    gc.collect()
            
            if end_token_seen:
                break
        
        # Final chunk: decode remaining tokens
        remaining_tokens = len(snac_buffer) - last_yield_position
        
        # Process final chunk even if smaller than window size
        if remaining_tokens >= 7:  # At least 1 frame (7 tokens)
            # If we have a full window, use it for better quality
            if remaining_tokens >= self.WINDOW_SIZE:
                window = snac_buffer[-self.WINDOW_SIZE:]
                # Use sliding window for consistency
                audio_bytes = await self.snac_decoder.decode_single_async(
                    window,
                    trim_warmup=False,
                    use_sliding_window=True
                )
                if audio_bytes:
                    # Decoder already returns middle 2048 samples when use_sliding_window=True
                    yield audio_bytes
            else:
                # Process whatever tokens we have left (must be divisible by 7)
                final_tokens = snac_buffer[last_yield_position:]
                # Ensure we have complete frames
                complete_frames = len(final_tokens) // 7
                if complete_frames > 0:
                    final_tokens = final_tokens[:complete_frames * 7]
                    audio_bytes = await self.snac_decoder.decode_single_async(
                        final_tokens, trim_warmup=False
                    )
                    if audio_bytes:
                        yield audio_bytes
        
        frames = len(snac_buffer) // 7
        duration = frames / 6.86
        print(f"Streamed {chunk_count} chunks (~{duration:.1f}s audio)")