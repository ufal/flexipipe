"""
SentencePiece-based segmenter for unknown languages.

Uses multilingual transformer models (XLM-RoBERTa, ByT5) with SentencePiece tokenization
to segment/tokenize text in completely unknown languages.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional
from pathlib import Path

# Suppress HuggingFace tokenizers parallelism warning
# This warning appears when tokenizers are used after forking, which is harmless
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ..doc import Document, Sentence, Token
from ..neural_backend import BackendManager, NeuralResult
from ..model_storage import get_backend_models_dir, setup_backend_environment
from ..backend_spec import BackendSpec


class SentencePieceSegmenter(BackendManager):
    """
    Segmenter that uses SentencePiece tokenization from multilingual transformer models.
    
    This is useful for completely unknown languages where no language-specific
    segmentation model is available. Uses XLM-RoBERTa or ByT5 tokenizers which
    are language-agnostic.
    """
    
    # Default models that work well for unknown languages
    DEFAULT_MODELS = {
        "xlm-roberta": "xlm-roberta-base",  # SentencePiece, 250k vocab
        "byt5": "google/byt5-base",  # Byte-level tokenization
        "default": "xlm-roberta-base",  # Default choice
    }
    
    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize SentencePiece segmenter.
        
        Args:
            model_name: Model name (e.g., 'xlm-roberta-base', 'google/byt5-base')
                       If None, uses 'xlm-roberta-base' as default
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        cache_dir = get_backend_models_dir("transformers", create=True)
        setup_backend_environment("transformers")
        
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "SentencePiece segmenter requires 'transformers'. "
                "Install it with: pip install transformers"
            ) from exc
        
        # Use default model if not specified
        if not model_name:
            model_name = self.DEFAULT_MODELS["default"]
        elif model_name in self.DEFAULT_MODELS:
            # Allow shortcuts like 'xlm-roberta' or 'byt5'
            model_name = self.DEFAULT_MODELS[model_name]
        
        self._model_name = model_name
        
        # Load tokenizer (we only need the tokenizer, not the full model)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                use_fast=True,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer from '{model_name}': {e}. "
                f"Make sure the model name is valid and accessible."
            ) from e
        
        if not tokenizer.is_fast:
            raise ValueError(
                f"SentencePiece segmenter requires a fast tokenizer. "
                f"Model '{model_name}' does not provide one."
            )
        
        self._tokenizer = tokenizer
    
    @property
    def supports_training(self) -> bool:
        return False
    
    def _segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences based on sentence-ending punctuation.
        
        This is a simple sentence segmenter that splits on .!? followed by whitespace
        or end of text.
        """
        if not text.strip():
            return []
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Pattern: sentence ending punctuation followed by whitespace or end
        # This matches: . ! ? followed by space or end of string
        pattern = r'([.!?]+)(?:\s+|$)'
        
        sentences = []
        last_pos = 0
        
        for match in re.finditer(pattern, text):
            # Extract sentence from last position to end of punctuation
            sentence_end = match.end()
            sentence_text = text[last_pos:sentence_end].strip()
            if sentence_text:
                sentences.append(sentence_text)
            last_pos = sentence_end
        
        # Add remaining text as final sentence (if any)
        if last_pos < len(text):
            remaining = text[last_pos:].strip()
            if remaining:
                sentences.append(remaining)
        
        # Fallback: if no sentences found, return entire text
        if not sentences:
            sentences.append(text)
        
        return sentences
    
    def _separate_punctuation(self, token_text: str) -> List[str]:
        """
        Separate punctuation from words.
        
        Returns a list of tokens: [word, punct1, punct2, ...] or [punct] if no word.
        """
        if not token_text:
            return []
        
        # Pattern to match: word characters (including Unicode letters) or punctuation
        # This will split "word;" into ["word", ";"]
        # Match word characters (including Unicode) or punctuation
        word_pattern = r'[\w\u0080-\uFFFF]+'  # Unicode word characters
        punct_pattern = r'[^\w\s\u0080-\uFFFF]+'  # Non-word, non-whitespace (punctuation)
        
        tokens = []
        i = 0
        
        while i < len(token_text):
            # Try to match a word first
            word_match = re.match(word_pattern, token_text[i:])
            if word_match:
                word = word_match.group(0)
                tokens.append(word)
                i += len(word)
            else:
                # Match punctuation
                punct_match = re.match(punct_pattern, token_text[i:])
                if punct_match:
                    punct = punct_match.group(0)
                    tokens.append(punct)
                    i += len(punct)
                else:
                    # Skip whitespace
                    i += 1
        
        return tokens if tokens else [token_text]
    
    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[dict] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
        pretokenize: bool = False,
    ) -> NeuralResult:
        """
        Segment/tokenize document using SentencePiece tokenization.
        
        By default, only performs sentence segmentation (splits text into sentences).
        If pretokenize=True, also tokenizes each sentence using SentencePiece tokenization
        and separates punctuation from words.
        
        Args:
            pretokenize: If True, also tokenize sentences. If False, only segment into sentences.
        """
        if document is None:
            raise ValueError("Document cannot be None")
        
        del overrides, preserve_pos_tags, components
        
        total_tokens = 0
        new_sentences = []
        
        # Process each sentence (may need to split into multiple sentences)
        for sentence in document.sentences:
            if not sentence.text:
                continue
            
            text = sentence.text.strip()
            if not text:
                continue
            
            # First, segment into sentences if needed
            # Check if this sentence contains multiple sentence endings
            sentence_texts = self._segment_sentences(text)
            
            # Process each sentence
            for sent_text in sentence_texts:
                if not sent_text.strip():
                    continue
                
                # If not pretokenizing, just create sentence without tokens
                if not pretokenize:
                    new_sentence = Sentence(id=str(len(new_sentences) + 1), text=sent_text, tokens=[])
                    new_sentences.append(new_sentence)
                    continue
                
                # Tokenize with offset mapping to preserve original text
                encoding = self._tokenizer(
                    sent_text,
                    return_tensors=None,  # Don't need tensors, just tokenization
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                    add_special_tokens=True,
                )
                
                offset_mapping = encoding.get("offset_mapping", [])
                special_tokens_mask = encoding.get("special_tokens_mask", [])
                
                if not offset_mapping:
                    # Fallback: use tokenizer's basic tokenization
                    tokens_list = self._tokenizer.tokenize(sent_text, add_special_tokens=False)
                    # Create tokens from tokenizer output (less accurate but works)
                    new_tokens = []
                    current_pos = 0
                    for idx, token_text in enumerate(tokens_list):
                        # Remove SentencePiece markers
                        clean_token = token_text.replace("▁", "").strip()
                        if not clean_token:
                            continue
                        
                        # Try to find the token in the original text
                        # This is approximate but better than nothing
                        if clean_token in sent_text[current_pos:]:
                            start = sent_text.find(clean_token, current_pos)
                            end = start + len(clean_token)
                            current_pos = end
                        else:
                            # If we can't find it, skip it (shouldn't happen often)
                            continue
                        
                        # Separate punctuation from word
                        separated = self._separate_punctuation(sent_text[start:end])
                        for sep_token in separated:
                            token = Token(
                                id=len(new_tokens) + 1,
                                form=sep_token,
                                upos="",
                                upos_confidence=0.0,
                            )
                            new_tokens.append(token)
                    
                    # Create new sentence with tokens
                    new_sentence = Sentence(id=str(len(new_sentences) + 1), text=sent_text, tokens=new_tokens)
                    new_sentences.append(new_sentence)
                    total_tokens += len(new_tokens)
                else:
                    # Use offset mapping for accurate tokenization
                    # Group subword tokens into words based on character offsets
                    # Key: only merge subwords that are part of the same word (no whitespace between them)
                    new_tokens = []
                    token_idx = 0
                    
                    i = 0
                    token_offsets = []  # Track (start, end) for each final token
                    
                    while i < len(offset_mapping):
                        offset = offset_mapping[i]
                        is_special = special_tokens_mask[i] if i < len(special_tokens_mask) else 0
                        
                        start, end = offset
                        
                        # Skip special tokens (CLS, SEP, PAD, etc.)
                        if is_special or start == end:
                            i += 1
                            continue
                        
                        # Start a new word with this token
                        word_start = start
                        word_end = end
                        j = i + 1
                        
                        # Look ahead to merge subwords that belong to the same word
                        # Only merge if tokens are adjacent (no gap) or if gap is only whitespace
                        while j < len(offset_mapping):
                            next_offset = offset_mapping[j]
                            next_is_special = special_tokens_mask[j] if j < len(special_tokens_mask) else 0
                            next_start, next_end = next_offset
                            
                            if next_is_special or next_start == next_end:
                                j += 1
                                continue
                            
                            # Check if next token is part of the same word
                            # For SentencePiece tokenizers, subwords of the same word are:
                            # 1. Adjacent (next_start == word_end) - no gap
                            # 2. Overlapping (next_start < word_end) - shouldn't happen but handle it
                            # 
                            # If there's ANY gap (even whitespace), it's a new word
                            # SentencePiece tokenizers don't insert gaps between subwords
                            
                            if next_start < word_end:
                                # Overlapping - merge them (shouldn't happen with proper tokenizers)
                                word_end = max(word_end, next_end)
                                j += 1
                            elif next_start == word_end:
                                # Adjacent - part of same word (subword tokenization)
                                word_end = max(word_end, next_end)
                                j += 1
                            else:
                                # There's a gap - this means a new word starts
                                # Even if it's just whitespace, it's a word boundary
                                break
                        
                        # Extract word text from original text
                        if word_start < len(sent_text) and word_end <= len(sent_text):
                            token_text = sent_text[word_start:word_end]
                        else:
                            i = j
                            continue
                        
                        if not token_text.strip():
                            i = j
                            continue
                        
                        # Separate punctuation from words
                        separated = self._separate_punctuation(token_text)
                        for sep_token in separated:
                            token = Token(
                                id=token_idx + 1,
                                form=sep_token,
                                upos="",
                                upos_confidence=0.0,
                            )
                            new_tokens.append(token)
                            token_offsets.append((word_start, word_end))  # Keep original offsets for space_after
                            token_idx += 1
                        
                        i = j
                    
                    # Set space_after based on actual text between tokens
                    for i in range(len(new_tokens) - 1):
                        if i < len(token_offsets) and i + 1 < len(token_offsets):
                            current_end = token_offsets[i][1]
                            next_start = token_offsets[i + 1][0]
                            if current_end < len(sent_text) and next_start <= len(sent_text):
                                between = sent_text[current_end:next_start]
                                # space_after is True if there's whitespace (but not just empty string)
                                new_tokens[i].space_after = bool(between and between.strip() == "" and between != "")
                            else:
                                new_tokens[i].space_after = True
                        else:
                            new_tokens[i].space_after = True
                    
                    if new_tokens:
                        new_tokens[-1].space_after = None
                    
                    # Create new sentence with tokens
                    new_sentence = Sentence(id=str(len(new_sentences) + 1), text=sent_text, tokens=new_tokens)
                    new_sentences.append(new_sentence)
                    total_tokens += len(new_tokens)
        
        # Replace document sentences with newly segmented and tokenized sentences
        document.sentences = new_sentences
        
        stats = {
            "model": self._model_name,
            "segmenter_type": "sentencepiece",
            "tokens": total_tokens,
        }
        return NeuralResult(document=document, stats=stats)
    
    def train(
        self,
        train_data: Document | List[Document] | Path,
        output_dir: Path,
        *,
        dev_data: Document | List[Document] | Path | None = None,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError("SentencePiece segmenter does not support training.")


def _create_sentencepiece_segmenter(
    *,
    model_name: Optional[str] = None,
    device: str = "cpu",
    **kwargs: Any,
) -> SentencePieceSegmenter:
    """Factory function for SentencePiece segmenter."""
    unexpected = set(kwargs) - {"download_model", "training"}
    if unexpected:
        raise ValueError(f"Unexpected SentencePiece segmenter arguments: {', '.join(sorted(unexpected))}")
    
    return SentencePieceSegmenter(
        model_name=model_name,
        device=device,
    )


# Backend specification for auto-discovery
spec = BackendSpec(
    name="sentencepiece",
    description="SentencePiece-based segmenter for unknown languages using multilingual transformer tokenizers (XLM-RoBERTa, ByT5)",
    factory=_create_sentencepiece_segmenter,
    supports_training=False,
    is_hidden=False,  # Visible in backend list
    install_instructions="Requires 'transformers' library. Install via: pip install transformers",
)

