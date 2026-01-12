"""
Improved Unicode-based tokenizer for flexipipe.

This tokenizer uses Unicode categories to properly handle:
- Combining marks (diacritics) - always attached to base characters
- Letters and numbers - grouped together
- Punctuation - separated from words
- Configurable handling of contractions and hyphenated compounds
"""

from __future__ import annotations

import unicodedata
import re
from typing import List, Optional, Tuple


def tokenize_unicode(
    text: str,
    *,
    keep_contractions: bool = True,
    keep_hyphenated: bool = True,
    keep_trailing_apostrophe: bool = True,
) -> List[str]:
    """
    Tokenize text using Unicode categories.
    
    This approach properly handles combining marks (diacritics) by always
    attaching them to their base characters, which is crucial for languages
    like Yoruba, Arabic, and many others.
    
    Args:
        text: Input text to tokenize
        keep_contractions: If True, keep contractions like "it's" as one token.
                          If False, split them: ["it", "'", "s"]
        keep_hyphenated: If True, keep hyphenated compounds like "state-of-the-art"
                        as one token. If False, split on hyphens.
        keep_trailing_apostrophe: If True, keep trailing apostrophes with words
                                  (e.g., "John's" stays together). If False, split.
    
    Returns:
        List of token strings
    """
    if not text:
        return []
    
    tokens = []
    buf = []
    in_contraction = False
    in_hyphenated = False
    
    def flush():
        """Flush current buffer to tokens."""
        if buf:
            tokens.append("".join(buf))
            buf.clear()
    
    def is_letter_or_number(cat: str) -> bool:
        """Check if category is letter or number."""
        return cat.startswith(("L", "N"))
    
    def is_combining_mark(cat: str) -> bool:
        """Check if category is combining mark (diacritic)."""
        return cat.startswith("M")
    
    def is_punctuation(cat: str) -> bool:
        """Check if category is punctuation."""
        return cat.startswith("P")
    
    def is_symbol(cat: str) -> bool:
        """Check if category is symbol."""
        return cat.startswith("S")
    
    def is_whitespace(cat: str) -> bool:
        """Check if category is whitespace."""
        return cat == "Zs" or cat.startswith("Z")
    
    i = 0
    while i < len(text):
        ch = text[i]
        cat = unicodedata.category(ch)
        
        # Handle combining marks - always attach to previous character
        if is_combining_mark(cat):
            if buf:
                # Attach to current buffer
                buf.append(ch)
            else:
                # No base character - this shouldn't happen in valid text,
                # but handle gracefully by creating a token
                buf.append(ch)
            i += 1
            continue
        
        # Handle letters and numbers
        if is_letter_or_number(cat):
            # Check if we need to flush (transition from non-word to word)
            # But don't flush if the previous char was:
            # - an apostrophe or hyphen (part of contraction/compound)
            # - a combining mark (part of the same word)
            # - a letter/number (part of the same word)
            if buf:
                prev_char = buf[-1]
                prev_cat = unicodedata.category(prev_char)
                # Don't flush if previous was apostrophe/hyphen/combining mark/letter/number
                should_flush = (
                    prev_char not in ("'", "'", "ʼ", "ʻ", "ʼ", "-") and 
                    not is_combining_mark(prev_cat) and
                    not is_letter_or_number(prev_cat)
                )
                if should_flush:
                    flush()
            buf.append(ch)
            i += 1
            continue
        
        # Handle apostrophes (for contractions)
        if ch in ("'", "'", "ʼ", "ʻ", "ʼ") and keep_contractions:
            # Check if this is part of a contraction
            # Look ahead and behind to see if we're between letters
            prev_is_letter = (
                buf and 
                is_letter_or_number(unicodedata.category(buf[-1]))
            )
            next_is_letter = (
                i + 1 < len(text) and
                is_letter_or_number(unicodedata.category(text[i + 1]))
            )
            
            if prev_is_letter and next_is_letter:
                # Part of contraction - keep with buffer
                buf.append(ch)
                i += 1
                continue
            elif prev_is_letter and keep_trailing_apostrophe:
                # Trailing apostrophe (possessive) - keep with buffer
                buf.append(ch)
                i += 1
                continue
        
        # Handle hyphens (for hyphenated compounds)
        if ch == "-":
            if keep_hyphenated:
                # Check if we're in a hyphenated compound
                prev_is_letter = (
                    buf and 
                    is_letter_or_number(unicodedata.category(buf[-1]))
                )
                next_is_letter = (
                    i + 1 < len(text) and
                    is_letter_or_number(unicodedata.category(text[i + 1]))
                )
                
                if prev_is_letter and next_is_letter:
                    # Part of hyphenated compound (e.g., "state-of-the-art") - keep with buffer
                    buf.append(ch)
                    i += 1
                    continue
            # Otherwise, treat as punctuation (will be flushed and added separately)
        
        # Handle whitespace - flush buffer
        if is_whitespace(cat):
            flush()
            i += 1
            continue
        
        # Handle punctuation and symbols - flush buffer, then add punctuation
        if is_punctuation(cat) or is_symbol(cat):
            flush()
            # Add punctuation as separate token
            tokens.append(ch)
            i += 1
            continue
        
        # Unknown category - flush and add as separate token
        flush()
        tokens.append(ch)
        i += 1
    
    # Flush any remaining buffer
    flush()
    
    # Filter out empty tokens
    return [t for t in tokens if t]


def segment_sentences(
    text: str,
    *,
    preserve_quotes: bool = True,
) -> List[str]:
    """
    Segment text into sentences.
    
    Args:
        text: Input text to segment
        preserve_quotes: If True, be smart about sentence boundaries inside quotes
    
    Returns:
        List of sentence strings
    """
    if not text.strip():
        return []
    
    sentences = []
    current_sentence = []
    in_double_quotes = False
    in_single_quotes = False
    
    i = 0
    while i < len(text):
        ch = text[i]
        
        # Track quote state
        if preserve_quotes:
            if ch == '"' and (i == 0 or text[i-1] != '\\'):
                in_double_quotes = not in_double_quotes
                current_sentence.append(ch)
                i += 1
                continue
            elif ch in ("'", "'") and (i == 0 or text[i-1] != '\\'):
                # Check if it's a quote or apostrophe
                prev_is_letter = i > 0 and (
                    unicodedata.category(text[i-1]).startswith("L") or
                    unicodedata.category(text[i-1]).startswith("N")
                )
                next_is_letter = i + 1 < len(text) and (
                    unicodedata.category(text[i+1]).startswith("L") or
                    unicodedata.category(text[i+1]).startswith("N")
                )
                
                if not (prev_is_letter or next_is_letter):
                    # It's a quote, not an apostrophe
                    in_single_quotes = not in_single_quotes
                    current_sentence.append(ch)
                    i += 1
                    continue
        
        # Check for sentence-ending punctuation
        if ch in ".!?" and not in_double_quotes and not in_single_quotes:
            current_sentence.append(ch)
            
            # Check if followed by whitespace or end of string
            if i + 1 >= len(text):
                # End of string - end sentence (period is part of this sentence)
                sentence_text = "".join(current_sentence).strip()
                if sentence_text:
                    sentences.append(sentence_text)
                current_sentence = []
            elif text[i+1] in " \t\n":
                # Whitespace after punctuation - end sentence (period is part of this sentence)
                sentence_text = "".join(current_sentence).strip()
                if sentence_text:
                    sentences.append(sentence_text)
                current_sentence = []
                # Skip whitespace
                i += 1
                while i < len(text) and text[i] in " \t\n":
                    i += 1
                continue
            elif i + 1 < len(text) and text[i+1] in '"\'':
                # Punctuation followed by quote - check if quote is closing
                if i + 2 < len(text) and text[i+2] in " \t\n":
                    # Quote then whitespace - end sentence (period is part of this sentence)
                    current_sentence.append(text[i+1])
                    sentence_text = "".join(current_sentence).strip()
                    if sentence_text:
                        sentences.append(sentence_text)
                    current_sentence = []
                    # Skip quote and whitespace
                    i += 2
                    while i < len(text) and text[i] in " \t\n":
                        i += 1
                    continue
        
        current_sentence.append(ch)
        i += 1
    
    # Add remaining text as final sentence
    # But skip if it's just punctuation (likely a duplicate from sentence splitting)
    if current_sentence:
        sentence_text = "".join(current_sentence).strip()
        # Only add if it's not just punctuation (which would be a duplicate period/exclamation/question mark)
        # This happens when sentence segmentation splits on punctuation but the punctuation
        # is already included in the previous sentence
        if sentence_text and not (len(sentence_text) == 1 and sentence_text in ".!?"):
            sentences.append(sentence_text)
        # If it's just punctuation and we have previous sentences, merge it with the last sentence
        elif sentence_text and len(sentence_text) == 1 and sentence_text in ".!?" and sentences:
            # The punctuation is already in the last sentence, so just skip this
            pass
    
    # Fallback: if no sentences found, return entire text as one sentence
    if not sentences:
        sentences.append(text.strip())
    
    return sentences

