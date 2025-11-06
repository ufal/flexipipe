"""
Vocabulary module for FlexiPipe.
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path

def find_similar_words(word: str, vocab: Dict[str, Dict], threshold: float = 0.7) -> List[Tuple[str, float]]:
    """Find similar words based on endings/beginnings.
    
    NOTE: Beginning-based matching is dangerous for lemmatization as it can incorrectly
    match words that share prefixes but are unrelated (e.g., "prometido" vs "recibir").
    We primarily rely on ending-based matching, which is more reliable for morphological patterns.
    """
    word_lower = word.lower()
    candidates = []
    
    # Check endings (last 3-6 characters) - this is the primary and most reliable method
    # Morphological patterns are typically suffix-based (inflections, derivations)
    for end_len in range(6, 2, -1):
        if len(word_lower) >= end_len:
            ending = word_lower[-end_len:]
            for vocab_word, vocab_data in vocab.items():
                if vocab_word.endswith(ending) and vocab_word != word_lower:
                    # Simple similarity: length difference and ending match
                    length_diff = abs(len(vocab_word) - len(word_lower)) / max(len(vocab_word), len(word_lower))
                    similarity = 1.0 - length_diff
                    if similarity >= threshold:
                        candidates.append((vocab_word, similarity))
    
    # Check beginnings (for some languages) - but with higher threshold and stricter matching
    # This is more dangerous as it can match unrelated words, so we're more conservative
    # Only use beginning matching if no ending matches were found
    if not candidates:
        for beg_len in range(5, 3, -1):  # Longer prefixes (4-5 chars) for better reliability
            if len(word_lower) >= beg_len:
                beginning = word_lower[:beg_len]
                for vocab_word, vocab_data in vocab.items():
                    # Require that the vocab word also starts with the same beginning
                    # AND has similar length (within 2 characters) to avoid wild matches
                    if vocab_word.startswith(beginning) and vocab_word != word_lower:
                        length_diff = abs(len(vocab_word) - len(word_lower))
                        if length_diff <= 2:  # Much stricter: only similar length words
                            similarity = 1.0 - (length_diff / max(len(vocab_word), len(word_lower)))
                            if similarity >= max(threshold, 0.8):  # Higher threshold for beginning matches
                                candidates.append((vocab_word, similarity))
    
    # Sort by similarity and return unique
    candidates.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_candidates = []
    for word, score in candidates:
        if word not in seen:
            seen.add(word)
            unique_candidates.append((word, score))
            if len(unique_candidates) >= 10:
                break
    
    return unique_candidates



