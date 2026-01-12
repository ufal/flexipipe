"""
Tokenization utilities for flexipipe.

This module handles pre-tokenization separately from segmentation.
Tokenization converts sentences (with text) into tokens.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .doc import Document
from .segmentation import (
    parse_segmenter_spec,
    backend_supports_segmentation,
    find_segmenter_for_language,
    get_default_segmenter_for_language,
    apply_segmentation,
)


def should_use_tokenization(
    backend_name: str,
    document: Document,
    *,
    tokenizer_spec: Optional[str] = None,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    auto_detect: bool = True,
) -> bool:
    """
    Determine if pre-tokenization should be used.
    
    Args:
        backend_name: Name of the main backend
        document: Input document
        tokenizer_spec: Explicit tokenizer specification (if provided)
        language: Language code
        auto_detect: If True, auto-detect need for tokenization
    
    Returns:
        True if tokenization should be used, False otherwise
    """
    # If tokenizer is explicitly specified, use it
    if tokenizer_spec is not None:  # Note: empty string means "disable", None means "auto"
        if tokenizer_spec == "":
            return False  # Explicitly disabled
        return True  # Explicitly specified
    
    # Special case: treetagger always needs tokenization (it doesn't tokenize itself)
    if backend_name.lower() == "treetagger":
        # Check if document already has tokens
        if any(sent.tokens for sent in document.sentences):
            return False  # Already tokenized
        # Check if document has sentences with text to tokenize
        if any(sent.text for sent in document.sentences):
            return True  # Need tokenization for treetagger
    
    # If auto-detection is disabled, don't use tokenization
    if not auto_detect:
        return False
    
    # Check registry for model requirements
    if model_name and backend_name:
        try:
            from .model_catalog import build_unified_catalog
            catalog = build_unified_catalog(use_cache=True, verbose=False)
            model_key = f"{backend_name}:{model_name}"
            model_entry = catalog.get(model_key)
            if model_entry:
                requires_tokenization = model_entry.get("requires_tokenization")
                if requires_tokenization is not None:
                    # Registry explicitly specifies whether tokenization is needed
                    return bool(requires_tokenization)
        except Exception:
            # If registry lookup fails, continue with other checks
            pass
    
    # Tokenization is opt-in: only auto-detect if explicitly needed
    # Check if backend supports tokenization/segmentation
    if backend_supports_segmentation(backend_name):
        return False  # Backend can handle it itself
    
    # Check if document already has tokens
    if any(sent.tokens for sent in document.sentences):
        return False  # Already tokenized
    
    # Check if document has sentences with text to tokenize
    if not any(sent.text for sent in document.sentences):
        return False  # No text to tokenize
    
    # Don't auto-detect tokenization - it's opt-in only
    # User must explicitly specify --tokenizer if they want pre-tokenization
    return False


def resolve_tokenizer_spec(
    tokenizer_spec: Optional[str],
    language: Optional[str],
    *,
    backend_name: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Resolve tokenizer specification to (backend, model) tuple.
    
    Args:
        tokenizer_spec: Explicit tokenizer spec (e.g., 'udpipe:yo' or 'udpipe' or '' to disable)
        language: Language code for auto-detection
        use_cache: Whether to use cached model catalog
        verbose: Whether to print verbose messages
    
    Returns:
        Tuple of (backend_name, model_name) or None if no tokenizer needed
    """
    # If explicit spec provided (and not empty string), parse it
    if tokenizer_spec and tokenizer_spec != "":
        backend, model = parse_segmenter_spec(tokenizer_spec)
        if verbose:
            model_str = f":{model}" if model else ""
            print(f"[flexipipe] Using explicit tokenizer: {backend}{model_str}", file=__import__('sys').stderr)
        return (backend, model)
    
    # If empty string, explicitly disabled
    if tokenizer_spec == "":
        if verbose:
            print("[flexipipe] Tokenization explicitly disabled", file=__import__('sys').stderr)
        return None
    
    # Check config for default tokenizer
    from .model_storage import get_default_tokenizer
    default_tokenizer = get_default_tokenizer()
    if default_tokenizer:
        backend, model = parse_segmenter_spec(default_tokenizer)
        if verbose:
            model_str = f":{model}" if model else ""
            print(f"[flexipipe] Using configured default tokenizer: {backend}{model_str}", file=__import__('sys').stderr)
        return (backend, model)
    
    # Check registry for preferred tokenizer
    if model_name and backend_name:
        try:
            from .model_catalog import build_unified_catalog
            catalog = build_unified_catalog(use_cache=use_cache, verbose=False)
            model_key = f"{backend_name}:{model_name}"
            model_entry = catalog.get(model_key)
            if model_entry:
                preferred_tokenizer = model_entry.get("preferred_tokenizer")
                if preferred_tokenizer:
                    # Registry specifies preferred tokenizer
                    backend, model = parse_segmenter_spec(preferred_tokenizer)
                    if verbose:
                        model_str = f":{model}" if model else ""
                        print(f"[flexipipe] Using registry-specified tokenizer for model '{model_name}': {backend}{model_str}", file=__import__('sys').stderr)
                    return (backend, model)
        except Exception:
            # If registry lookup fails, continue with other checks
            pass
    
    # Auto-detect: try to find language-specific tokenizer
    tokenizer = find_segmenter_for_language(language, use_cache=use_cache)
    if tokenizer:
        if verbose:
            backend, model = tokenizer
            model_str = f":{model}" if model else ""
            print(f"[flexipipe] Auto-detected tokenizer for '{language}': {backend}{model_str}", file=__import__('sys').stderr)
        return tokenizer
    
    # No language-specific tokenizer found - use best default tokenizer
    # (sentencepiece if available, otherwise flexitag)
    from .segmentation import _get_best_default_segmenter
    default_tokenizer = _get_best_default_segmenter()  # Same logic as segmenter
    if verbose:
        print(f"[flexipipe] No language-specific tokenizer found for '{language}', using {default_tokenizer} (language-independent)", file=__import__('sys').stderr)
    return (default_tokenizer, None)


def apply_tokenization(
    document: Document,
    tokenizer: Tuple[str, Optional[str]],
    *,
    language: Optional[str] = None,
    verbose: bool = False,
) -> Document:
    """
    Apply pre-tokenization to a document using the specified tokenizer.
    
    Args:
        document: Input document (should have sentences with text but no tokens)
        tokenizer: Tuple of (backend_name, model_name) or tokenizer spec string
        language: Language code for tokenizer (if different from document language)
        verbose: Whether to print verbose messages
    
    Returns:
        Document with tokens added from tokenization
    """
    # Reuse apply_segmentation but with pretokenize=True
    return apply_segmentation(
        document,
        tokenizer,
        language=language,
        verbose=verbose,
        pretokenize=True,  # Force tokenization
    )


def maybe_apply_tokenization(
    document: Document,
    backend_name: str,
    *,
    tokenizer_spec: Optional[str] = None,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Tuple[Document, bool]:
    """
    Apply tokenization if needed, returning the (possibly tokenized) document and whether tokenization was applied.
    
    This is the main entry point for tokenization in the pipeline. It:
    1. Determines if tokenization is needed
    2. Resolves the tokenizer to use
    3. Applies tokenization if needed
    
    Args:
        document: Input document
        backend_name: Name of the main backend
        tokenizer_spec: Explicit tokenizer specification (e.g., 'udpipe:yo' or '' to disable)
        language: Language code
        use_cache: Whether to use cached model catalog
        verbose: Whether to print verbose messages
    
    Returns:
        Tuple of (document, tokenization_applied)
        - document: Document (possibly with tokens added from tokenization)
        - tokenization_applied: True if tokenization was applied, False otherwise
    """
    # Check if tokenization is needed
    if not should_use_tokenization(
        backend_name,
        document,
        tokenizer_spec=tokenizer_spec,
        language=language,
        model_name=model_name,
        auto_detect=True,
    ):
        return document, False
    
    # Resolve tokenizer specification
    resolved_tokenizer = resolve_tokenizer_spec(
        tokenizer_spec,
        language,
        backend_name=backend_name,
        model_name=model_name,
        use_cache=use_cache,
        verbose=verbose,
    )
    
    if not resolved_tokenizer:
        return document, False
    
    # Apply tokenization
    tokenized_doc = apply_tokenization(
        document,
        resolved_tokenizer,
        language=language,
        verbose=verbose,
    )
    
    return tokenized_doc, True

