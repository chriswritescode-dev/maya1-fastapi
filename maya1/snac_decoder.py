import torch
import numpy as np
import asyncio
import time
from typing import List, Optional, Tuple
from snac import SNAC

from .constants import (
    CODE_END_TOKEN_ID,
    CODE_TOKEN_OFFSET,
    SNAC_MODEL_NAME,
    SNAC_TOKENS_PER_FRAME,
    SNAC_BATCH_SIZE,
    SNAC_BATCH_TIMEOUT_MS,
)


class SNACDecoder:
    """
    SNAC Decoder for maya1.
    Unpacks 7-token SNAC frames and decodes to audio waveforms.
    Unpacking logic is the EXACT INVERSE of training preprocessing.
    Supports async batching for concurrent requests.
    CRITICAL: Any mismatch in unpacking will produce garbage audio.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        compile_decoder: bool = False,
        enable_batching: bool = False,
        max_batch_size: int = SNAC_BATCH_SIZE,
        batch_timeout_ms: int = SNAC_BATCH_TIMEOUT_MS,
    ):
        """
        Initialize SNAC decoder.
        
        Args:
            device: Device for SNAC model (cuda/cpu)
            compile_decoder: Use torch.compile for speedup
            enable_batching: Enable async batching
            max_batch_size: Max sequences to batch together
            batch_timeout_ms: Max wait time before processing batch
        """
        self.device = device
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        self.total_processed = 0
        self.total_batches = 0
        self.peak_mem_mb = 0.0
        
        print(f"Loading SNAC 24kHz model to {device}...")
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval().to(device)
        
        if compile_decoder:
            print(f"Compiling SNAC decoder with torch.compile...")
            self._compile_model()
        
        # Batching infrastructure
        if enable_batching:
            max_queue_size = max_batch_size * 2  # Reduced from 4x to 2x to save memory
            self.request_queue = asyncio.Queue(maxsize=max_queue_size)
            self.batch_processor_task = None
            self._running = False
            print(f"Batching enabled (max_batch={max_batch_size}, timeout={batch_timeout_ms}ms, max_queue={max_queue_size})")
        
        print(f"SNAC decoder initialized")
    
    def _compile_model(self):
        """Compile SNAC decoder with torch.compile"""
        for frames in [4, 16, 32]:
            dummy_codes = [
                torch.randint(0, 4096, (1, frames), device=self.device),
                torch.randint(0, 4096, (1, frames * 2), device=self.device),
                torch.randint(0, 4096, (1, frames * 4), device=self.device),
            ]
            with torch.inference_mode():
                z_q = self.snac_model.quantizer.from_codes(dummy_codes)
                _ = self.snac_model.decoder(z_q)
        
        self.snac_model.decoder = torch.compile(
            self.snac_model.decoder,
            mode="max-autotune"
        )
        self.snac_model.quantizer = torch.compile(
            self.snac_model.quantizer,
            mode="reduce-overhead"
        )
        
        print(f"SNAC decoder compiled")
    
    def _get_gpu_memory_mb(self) -> float:
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def _log_memory(self, tag: str = ""):
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            print(f"[GPU Memory {tag}] Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
            if allocated > self.peak_mem_mb:
                self.peak_mem_mb = allocated
    
    def get_stats(self) -> dict:
        return {
            "total_processed": self.total_processed,
            "total_batches": self.total_batches,
            "peak_memory_mb": self.peak_mem_mb,
            "current_memory_mb": self._get_gpu_memory_mb(),
        }
    
    def unpack_snac_from_7(self, vocab_ids: List[int]) -> List[List[int]]:
        """
        Unpack 7-token SNAC frames to 3 hierarchical levels.
        
        This is the EXACT INVERSE of the training preprocessing function
        `pack_snac_to_7_and_offset()`.
        
        Frame structure:
        [slot0, slot1, slot2, slot3, slot4, slot5, slot6]
        
        Unpacking:
        - slot0: L1[i]
        - slot1: L2[2*i]      (even index)
        - slot2: L3[4*i + 0]
        - slot3: L3[4*i + 1]
        - slot4: L2[2*i + 1]  (odd index)
        - slot5: L3[4*i + 2]
        - slot6: L3[4*i + 3]
        
        Args:
            vocab_ids: List of SNAC token IDs (128266-156937)
                       Must be divisible by 7
        
        Returns:
            [L1, L2, L3] where:
                L1: n elements (coarse level)
                L2: 2n elements (medium level)
                L3: 4n elements (fine level)
        """
        # Strip EOS token if present
        if vocab_ids and vocab_ids[-1] == CODE_END_TOKEN_ID:
            vocab_ids = vocab_ids[:-1]
        
        # Ensure complete frames (divisible by 7)
        frames = len(vocab_ids) // SNAC_TOKENS_PER_FRAME
        vocab_ids = vocab_ids[:frames * SNAC_TOKENS_PER_FRAME]
        
        if frames == 0:
            return [[], [], []]
        
        l1, l2, l3 = [], [], []
        
        for i in range(frames):
            # Extract 7 slots for this frame
            slots = vocab_ids[i*7:(i+1)*7]
            
            # Subtract offset (128266) and mod 4096 to get original codes
            # Each level uses 4096 codes (0-4095)
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,  # Even index
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,  # Odd index
            ])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ])
        
        return [l1, l2, l3]
    
    def decode(
        self, 
        snac_tokens: List[int], 
        trim_warmup: bool = True, 
        trim_amount: Optional[int] = None,
        use_sliding_window: bool = False
    ) -> Optional[np.ndarray]:
        """
        Decode SNAC tokens to audio waveform.
        
        Args:
            snac_tokens: List of SNAC token IDs (7*n tokens)
            trim_warmup: Whether to trim SNAC warmup samples (default: True)
            trim_amount: Number of samples to trim (default: 2048 for first chunk, 0 for others)
                        Can be set to a smaller value (e.g., 512) for intermediate chunks
            use_sliding_window: If True, only return middle 2048 samples (for sliding window streaming)
        
        Returns:
            Audio waveform as numpy array (float32, 24kHz mono)
            Shape: (samples,)
            Returns None if not enough tokens
        """
        if len(snac_tokens) < SNAC_TOKENS_PER_FRAME:
            print(f"Not enough SNAC tokens: {len(snac_tokens)} < {SNAC_TOKENS_PER_FRAME}")
            return None
        
        levels = self.unpack_snac_from_7(snac_tokens)
        
        if not levels[0]:
            return None
        
        codes = None
        z_q = None
        audio_tensor = None
        
        try:
            codes = [
                torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0)
                for level in levels
            ]
            
            z_q = self.snac_model.quantizer.from_codes(codes)
            
            # Delete codes immediately after use
            del codes
            codes = None
            
            audio_tensor = self.snac_model.decoder(z_q)
            
            # Delete z_q immediately after use
            del z_q
            z_q = None
            
            audio = audio_tensor[0, 0].detach().cpu().numpy()
            
            # Delete audio_tensor immediately after extracting audio
            del audio_tensor
            audio_tensor = None
            
        finally:
            # Ensure cleanup on error
            if audio_tensor is not None:
                del audio_tensor
            if z_q is not None:
                del z_q
            if codes is not None:
                for code in codes:
                    del code
                del codes
            
            # Single cache clear at the end
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        if use_sliding_window:
            if len(audio) >= 4096:
                audio = audio[2048:4096]
        else:
            if trim_warmup:
                if trim_amount is None:
                    trim_amount = 2048
                
                if len(audio) > trim_amount:
                    audio = audio[trim_amount:]
        
        return audio
    
    def decode_to_bytes(
        self, 
        snac_tokens: List[int], 
        trim_warmup: bool = True,
        use_sliding_window: bool = False
    ) -> Optional[bytes]:
        """
        Decode SNAC tokens to audio bytes (int16 PCM).
        
        Args:
            snac_tokens: List of SNAC token IDs
            trim_warmup: Whether to trim SNAC warmup samples (default: True)
            use_sliding_window: If True, only return middle 2048 samples (for sliding window streaming)
        
        Returns:
            Audio as bytes (int16 PCM, 24kHz mono)
            Returns None if decode fails
        """
        audio = self.decode(snac_tokens, trim_warmup=trim_warmup, use_sliding_window=use_sliding_window)
        
        if audio is None:
            return None
        
        # Convert float32 to int16 PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        return audio_int16.tobytes()
    
    def validate_tokens(self, snac_tokens: List[int]) -> bool:
        """
        Validate SNAC tokens before decoding.
        Args:
            snac_tokens: List of SNAC token IDs
        Returns:
            True if valid, False otherwise
        """
        # Check minimum length
        if len(snac_tokens) < SNAC_TOKENS_PER_FRAME:
            print(f"Too few tokens: {len(snac_tokens)}")
            return False
        
        # Check divisibility by 7
        if len(snac_tokens) % SNAC_TOKENS_PER_FRAME != 0:
            print(f"  Warning: Token count {len(snac_tokens)} not divisible by 7")
            print(f"   Will truncate to {(len(snac_tokens) // 7) * 7}")
        
        # Check token range
        for i, token_id in enumerate(snac_tokens):
            if token_id < CODE_TOKEN_OFFSET or token_id > 156937:
                print(f" Invalid token at position {i}: {token_id}")
                print(f"   Expected range: [{CODE_TOKEN_OFFSET}, 156937]")
                return False
        
        return True
    
    # ========== Async Batching Methods ==========
    
    @property
    def is_running(self) -> bool:
        """Check if batch processor is running."""
        return self._running if self.enable_batching else False
    
    async def start_batch_processor(self):
        """Start the background batch processor task."""
        if not self.enable_batching:
            return
        
        if self._running:
            print("Batch processor already running")
            return
        
        self._running = True
        self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        print("Batch processor started")
    
    async def stop_batch_processor(self):
        """Stop the background batch processor task."""
        if not self.enable_batching:
            return
        
        if not self._running:
            return
        
        self._running = False
        
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        print("Batch processor stopped")
    
    async def decode_single_async(
        self, 
        snac_tokens: List[int], 
        trim_warmup: bool = True,
        use_sliding_window: bool = False
    ) -> Optional[bytes]:
        """
        Async decode for batching support.
        
        Queues the request and waits for batched processing.
        
        Args:
            snac_tokens: List of SNAC token IDs
            trim_warmup: Whether to trim SNAC warmup samples (default: True)
            use_sliding_window: If True, only return middle 2048 samples (for sliding window streaming)
        
        Returns:
            Audio bytes or None if decode fails
        """
        if not self.enable_batching:
            # Fallback to synchronous decode
            return self.decode_to_bytes(snac_tokens, trim_warmup=trim_warmup, use_sliding_window=use_sliding_window)
        
        # Create future for result
        result_future = asyncio.Future()
        
        # Add to queue (include trim_warmup and sliding_window flags)
        await self.request_queue.put((snac_tokens, trim_warmup, use_sliding_window, result_future))
        
        # Wait for result
        return await result_future
    
    async def _batch_processor_loop(self):
        """Background task that processes batched decode requests."""
        while self._running:
            try:
                # Collect batch
                batch = await self._collect_batch()
                
                if not batch:
                    continue
                
                # Process batch
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Batch processor error: {e}")
                import traceback
                traceback.print_exc()
    
    async def _collect_batch(self) -> List[Tuple[List[int], bool, bool, asyncio.Future]]:
        """
        Collect requests into a batch.
        Waits for timeout or until batch is full.
        Returns:
            List of (tokens, trim_warmup, use_sliding_window, future) tuples
        """
        batch = []
        timeout_sec = self.batch_timeout_ms / 1000.0
        
        try:
            # Wait for first request (blocking)
            first_item = await asyncio.wait_for(
                self.request_queue.get(),
                timeout=timeout_sec
            )
            batch.append(first_item)
            
            # Collect more requests (non-blocking)
            while len(batch) < self.max_batch_size:
                try:
                    item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=timeout_sec
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break  # Timeout reached, process what we have
        
        except asyncio.TimeoutError:
            # No requests in timeout period
            pass
        
        return batch
    
    async def _process_batch(self, batch: List[Tuple[List[int], bool, bool, asyncio.Future]]):
        """
        Process a batch of decode requests.
        Args:
            batch: List of (tokens, trim_warmup, use_sliding_window, future) tuples
        """
        if not batch:
            return
        
        start_time = time.time()
        self.total_batches += 1
        
        # Extract components
        token_sequences = [item[0] for item in batch]
        trim_warmup_flags = [item[1] for item in batch]
        sliding_window_flags = [item[2] for item in batch]
        futures = [item[3] for item in batch]
        
        lengths = [len(tokens) for tokens in token_sequences]
        can_batch_efficiently = len(set(lengths)) == 1
        
        self._log_memory(f"Batch start (size={len(batch)})")
        
        if can_batch_efficiently and len(batch) > 1:
            # Efficient batching: all same length
            try:
                audio_bytes_list = await self._decode_batch_same_length(
                    token_sequences, trim_warmup_flags, sliding_window_flags
                )
                
                # Set results and immediately clear references
                for future, audio_bytes in zip(futures, audio_bytes_list):
                    if not future.done():
                        future.set_result(audio_bytes)
                
                self.total_processed += len(batch)
                
                # Clear the list after setting results
                del audio_bytes_list
            
            except Exception as e:
                # Set exceptions
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
        else:
            # Sequential decode (different lengths or single item)
            for tokens, trim_warmup, use_sliding_window, future in batch:
                try:
                    audio_bytes = self.decode_to_bytes(
                        tokens, trim_warmup=trim_warmup, use_sliding_window=use_sliding_window
                    )
                    if not future.done():
                        future.set_result(audio_bytes)
                    self.total_processed += 1
                    
                    # Clear audio_bytes immediately after setting result
                    del audio_bytes
                    
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
        
        # Clear batch data
        del token_sequences
        del trim_warmup_flags
        del sliding_window_flags
        del futures
        del batch
        
        elapsed = (time.time() - start_time) * 1000
        self._log_memory(f"Batch end (took {elapsed:.1f}ms)")
        
        # Single cache clear at the end
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    async def _decode_batch_same_length(
        self, 
        token_sequences: List[List[int]], 
        trim_warmup_flags: List[bool],
        sliding_window_flags: List[bool]
    ):
        """
        Decode multiple sequences with same length in parallel.
        Optimized for minimal memory usage.
        
        Args:
            token_sequences: List of token sequences (all same length)
            trim_warmup_flags: List of trim_warmup flags for each sequence
            sliding_window_flags: List of use_sliding_window flags for each sequence
        
        Returns:
            List of audio bytes
        """
        if not token_sequences:
            return []
        
        # Unpack sequences one at a time to avoid holding all in memory
        batch_size = len(token_sequences)
        audio_bytes_list = [None] * batch_size
        
        # Find valid sequences and their unpacked data
        valid_indices = []
        valid_unpacked = []
        
        for i, tokens in enumerate(token_sequences):
            unpacked = self.unpack_snac_from_7(tokens)
            if unpacked[0]:  # Check if L1 is not empty
                valid_indices.append(i)
                valid_unpacked.append(unpacked)
        
        if not valid_indices:
            return audio_bytes_list
        
        codes = None
        z_q = None
        audio_batch = None
        
        try:
            # Create tensors more efficiently - pre-allocate and fill
            valid_batch_size = len(valid_indices)
            codes = []
            
            for level_idx in range(3):
                # Get the length of this level from the first valid sequence
                level_len = len(valid_unpacked[0][level_idx])
                
                # Pre-allocate tensor for this level
                level_tensor = torch.empty(
                    (valid_batch_size, level_len), 
                    dtype=torch.long, 
                    device=self.device
                )
                
                # Fill tensor in-place
                for batch_idx, unpacked in enumerate(valid_unpacked):
                    level_data = torch.tensor(
                        unpacked[level_idx], 
                        dtype=torch.long, 
                        device=self.device
                    )
                    level_tensor[batch_idx] = level_data
                    del level_data  # Immediate cleanup
                
                codes.append(level_tensor)
            
            # Clear unpacked data immediately
            del valid_unpacked
            
            # Process through model
            z_q = self.snac_model.quantizer.from_codes(codes)
            
            # Delete codes immediately after use
            del codes
            codes = None
            
            audio_batch = self.snac_model.decoder(z_q)
            
            # Delete z_q immediately after use
            del z_q
            z_q = None
            
            # Process audio outputs without unnecessary copies
            for batch_idx, orig_idx in enumerate(valid_indices):
                # Get audio without copy
                audio = audio_batch[batch_idx, 0].detach().cpu().numpy()
                
                # Apply trimming based on flags (without intermediate copies)
                if sliding_window_flags[orig_idx]:
                    if len(audio) >= 4096:
                        # Direct slice to int16
                        audio_slice = audio[2048:4096]
                        audio_int16 = (audio_slice * 32767).astype(np.int16)
                    else:
                        audio_int16 = (audio * 32767).astype(np.int16)
                else:
                    if trim_warmup_flags[orig_idx] and len(audio) > 2048:
                        # Direct slice to int16
                        audio_slice = audio[2048:]
                        audio_int16 = (audio_slice * 32767).astype(np.int16)
                    else:
                        audio_int16 = (audio * 32767).astype(np.int16)
                
                audio_bytes_list[orig_idx] = audio_int16.tobytes()
                
                # Immediate cleanup
                del audio_int16
                del audio
            
            # Clean up audio batch
            del audio_batch
            audio_batch = None
        
        finally:
            # Ensure cleanup even on error
            if audio_batch is not None:
                del audio_batch
            if z_q is not None:
                del z_q
            if codes is not None:
                del codes
            
            # Single cache clear at the end
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return audio_bytes_list
