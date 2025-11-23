"""
Text Chunking Utilities for Maya1 TTS
Automatically splits large text inputs into manageable chunks.
"""

import re
from typing import List


def split_text_into_chunks(text: str, max_chunk_size: int = 1800) -> List[str]:
    """
    Split text into chunks while preserving sentence boundaries.
    
    Args:
        text: Input text to split
        max_chunk_size: Maximum characters per chunk (leave room for emotion tags)
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences, respecting punctuation
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
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Sentence itself is too long, force split by words
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > max_chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            # Single word is too long, force split
                            while len(word) > max_chunk_size:
                                chunks.append(word[:max_chunk_size])
                                word = word[max_chunk_size:]
                            temp_chunk = word
                    else:
                        if temp_chunk:
                            temp_chunk += " " + word
                        else:
                            temp_chunk = word
                current_chunk = temp_chunk
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def estimate_audio_duration(text: str) -> float:
    """
    Estimate audio duration in seconds based on text length.
    
    Args:
        text: Input text
    
    Returns:
        Estimated duration in seconds
    """
    # Average reading speed: ~150 words per minute
    # Average word length: ~5 characters
    words_per_minute = 150
    chars_per_word = 5
    
    word_count = len(text.split())
    estimated_minutes = word_count / words_per_minute
    return estimated_minutes * 60


def get_chunk_info(text: str, max_chunk_size: int = 1800) -> dict:
    """
    Get information about text chunking.
    
    Args:
        text: Input text
        max_chunk_size: Maximum characters per chunk
    
    Returns:
        Dictionary with chunking information
    """
    chunks = split_text_into_chunks(text, max_chunk_size)
    total_duration = estimate_audio_duration(text)
    
    return {
        "original_length": len(text),
        "chunk_count": len(chunks),
        "max_chunk_size": max_chunk_size,
        "estimated_duration_seconds": total_duration,
        "chunk_lengths": [len(chunk) for chunk in chunks],
        "chunks": chunks
    }