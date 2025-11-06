"""
Maya1 Generation Pipeline
End-to-end pipeline for TTS generation (non-streaming).
"""

import asyncio
from typing import Optional, List
from vllm import SamplingParams

from .constants import (
    CODE_END_TOKEN_ID,
    CODE_START_TOKEN_ID,
    SNAC_MIN_ID,
    SNAC_MAX_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
)


class Maya1Pipeline:
    """End-to-end TTS pipeline for Maya1."""
    
    def __init__(self, model, prompt_builder, snac_decoder):
        """
        Initialize pipeline.
        Args:
            model: Maya1Model instance
            prompt_builder: Maya1PromptBuilder instance
            snac_decoder: SNACDecoder instance
        """
        self.model = model
        self.prompt_builder = prompt_builder
        self.snac_decoder = snac_decoder
        print(f"✅ Maya1Pipeline initialized")
    
    async def generate_speech(
        self,
        description: str,
        text: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: Optional[int] = None,
    ) -> Optional[bytes]:
        """
        Generate speech audio (non-streaming).
        Args:
            description: Voice description
            text: Text to synthesize (may include <emotion> tags)
            temperature: Sampling temperature
            top_p: Nucleus sampling
            max_tokens: Max SNAC tokens to generate
            repetition_penalty: Prevent loops
            seed: Random seed for reproducibility
        
        Returns:
            Audio bytes (int16 PCM, 24kHz mono) or None if failed
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
        
        # Generate tokens
        outputs = await self.model.generate(prompt, sampling_params)
        
        if not outputs or len(outputs) == 0:
            return None
        
        output = outputs[0]
        generated_token_ids = output.outputs[0].token_ids
        
        # Extract SNAC codes
        snac_codes = self._extract_snac_codes(generated_token_ids)
        
        if not snac_codes:
            return None
        
        # Decode to audio
        audio_bytes = await self.snac_decoder.decode_single_async(snac_codes)
        
        if audio_bytes:
            frames = len(snac_codes) // 7
            duration_sec = frames / 6.86
            print(f" Generated {frames} frames (~{duration_sec:.1f}s audio)")
        
        return audio_bytes
    
    def _extract_snac_codes(self, token_ids: List[int]) -> List[int]:
        # Find SOS and EOS positions
        try:
            sos_idx = token_ids.index(CODE_START_TOKEN_ID)
        except ValueError:
            sos_idx = -1
        
        try:
            eos_idx = token_ids.index(CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx = len(token_ids)
        
        # Extract tokens between SOS and EOS
        if sos_idx >= 0:
            snac_tokens = token_ids[sos_idx + 1:eos_idx]
        else:
            # If no SOS found, take everything before EOS
            snac_tokens = token_ids[:eos_idx]
        
        # Filter to only valid SNAC token IDs
        snac_codes = [
            token_id for token_id in snac_tokens
            if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
        ]
        
        return snac_codes
