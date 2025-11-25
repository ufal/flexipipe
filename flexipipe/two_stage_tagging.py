"""
Two-stage tagging workflow: POS tagging first, then parsing with preserved POS tags.

This module provides utilities for the common NLP workflow:
1. Tag POS tags (using flexitag or neural backend)
2. Manually correct POS tags if needed
3. Run dependency parsing with the corrected POS tags for enhanced accuracy

This is especially useful with:
- Transformers.adapters: Can pass POS tags as input features to parser adapters
- SpaCy: Can explicitly set POS tags before running the parser component
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .doc import Document
from .engine import FlexitagFallback, FlexitagResult
from .neural_backend import BackendManager, NeuralResult


def tag_pos_then_parse(
    document: Document,
    *,
    pos_tagger: Union[FlexitagFallback, BackendManager],
    parser: Optional[BackendManager] = None,
    pos_model_path: Optional[str] = None,
    parser_model_path: Optional[str] = None,
    preserve_pos: bool = True,
) -> Union[FlexitagResult, NeuralResult]:
    """
    Two-stage tagging: POS tagging first, then parsing with preserved POS tags.
    
    Args:
        document: Input document
        pos_tagger: Tagger to use for POS tagging (flexitag or neural)
        parser: Optional parser to use (if None, uses pos_tagger for parsing too)
        pos_model_path: Path to POS tagging model (if using flexitag)
        parser_model_path: Path to parser model (if different from POS model)
        preserve_pos: If True, preserve POS tags when running parser
    
    Returns:
        Tagged document with both POS tags and dependency parse
    
    Example:
        >>> from flexipipe.two_stage_tagging import tag_pos_then_parse
        >>> from flexipipe.engine import FlexitagFallback
        >>> from flexipipe.neural_backend import SpacyBackend
        >>> 
        >>> # Stage 1: POS tagging with flexitag
        >>> pos_tagger = FlexitagFallback("models/pos_model.json")
        >>> pos_result = pos_tagger.tag(document)
        >>> 
        >>> # (Optionally manually correct POS tags in pos_result.document)
        >>> 
        >>> # Stage 2: Parsing with SpaCy, preserving POS tags
        >>> parser = SpacyBackend.from_pretrained("en_core_web_sm")
        >>> final_result = tag_pos_then_parse(
        ...     pos_result.document,
        ...     pos_tagger=pos_tagger,
        ...     parser=parser,
        ...     preserve_pos=True
        ... )
    """
    # Stage 1: POS tagging
    if isinstance(pos_tagger, FlexitagFallback):
        pos_result = pos_tagger.tag(document)
        pos_doc = pos_result.document
    elif isinstance(pos_tagger, BackendManager):
        # Use neural backend for POS tagging only (disable parser)
        pos_result = pos_tagger.tag(document, components=["tagger", "lemmatizer"])
        pos_doc = pos_result.document
    else:
        raise TypeError(f"Unsupported pos_tagger type: {type(pos_tagger)}")
    
    # Stage 2: Parsing with preserved POS tags
    if parser is None:
        # Use the same tagger for parsing
        parser = pos_tagger
    
    if isinstance(parser, BackendManager):
        # Use neural backend for parsing, preserving POS tags
        parse_result = parser.tag(
            pos_doc,
            preserve_pos_tags=preserve_pos,
            components=["parser"]  # Only run parser component
        )
        return parse_result
    elif isinstance(parser, FlexitagFallback):
        # Flexitag doesn't have separate parsing, but it respects existing tags
        # Just run it again - it will preserve existing POS tags
        parse_result = parser.tag(pos_doc)
        return parse_result
    else:
        raise TypeError(f"Unsupported parser type: {type(parser)}")


def tag_with_flexitag_then_parse_with_spacy(
    document: Document,
    pos_model_path: str,
    spacy_model_name: str,
    *,
    preserve_pos: bool = True
) -> NeuralResult:
    """
    Convenience function: POS tag with flexitag, then parse with SpaCy.
    
    This is a common workflow where flexitag provides accurate POS tags
    (especially for historical/domain-specific texts), and SpaCy provides
    high-quality dependency parsing.
    
    Args:
        document: Input document
        pos_model_path: Path to flexitag POS model
        spacy_model_name: SpaCy model name (e.g., "en_core_web_sm")
        preserve_pos: If True, preserve flexitag POS tags when parsing
    
    Returns:
        Document with flexitag POS tags and SpaCy dependency parse
    """
    from .spacy_backend import SpacyBackend
    
    # Stage 1: POS tagging with flexitag
    pos_tagger = FlexitagFallback(pos_model_path)
    pos_result = pos_tagger.tag(document)
    
    # Stage 2: Parsing with SpaCy, preserving POS tags
    parser = SpacyBackend.from_pretrained(spacy_model_name)
    parse_result = parser.tag(
        pos_result.document,
        preserve_pos_tags=preserve_pos,
        components=["parser"]  # Only run parser, skip tagger
    )
    
    return parse_result


def tag_with_spacy_then_parse_with_adapters(
    document: Document,
    spacy_model_name: str,
    adapter_name: str,
    *,
    preserve_pos: bool = True
) -> NeuralResult:
    """
    Two-stage tagging with SpaCy POS and Transformers.adapters parser.
    
    This workflow uses:
    1. SpaCy for fast, accurate POS tagging
    2. Transformers.adapters for high-quality dependency parsing with feature passing
    
    Args:
        document: Input document
        spacy_model_name: SpaCy model for POS tagging
        adapter_name: Transformers adapter name for parsing
        preserve_pos: If True, pass POS tags as features to adapter
    
    Returns:
        Document with SpaCy POS tags and adapter-based dependency parse
    
    Note:
        This requires the Transformers backend to be implemented.
    """
    from .spacy_backend import SpacyBackend
    from .backends.transformers import HuggingFaceTransformersBackend
    
    # Stage 1: POS tagging with SpaCy
    pos_tagger = SpacyBackend.from_pretrained(spacy_model_name)
    pos_result = pos_tagger.tag(document, components=["tagger", "lemmatizer"])
    
    # Stage 2: Parsing with Transformers.adapters
    # The adapter will receive POS tags as input features
    parser = HuggingFaceTransformersBackend(
        model_name="bert-base-multilingual-cased",  # Base model
        adapter_name=adapter_name
    )
    parse_result = parser.tag(
        pos_result.document,
        preserve_pos_tags=preserve_pos,
        components=["parser"]
    )
    
    return parse_result

