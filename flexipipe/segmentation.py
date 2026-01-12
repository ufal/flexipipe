"""
Pre-segmentation and pre-tokenization support for flexipipe.

This module provides a unified interface for pre-segmenting/tokenizing text
before passing it to backends that don't support segmentation themselves.
"""

from __future__ import annotations

from typing import Optional, Tuple
from .doc import Document
from .backend_registry import create_backend, get_backend_info
from .neural_backend import BackendManager, NeuralResult


def parse_segmenter_spec(spec: str) -> Tuple[str, Optional[str]]:
    """
    Parse segmenter specification in format 'backend[:model]'.
    
    Examples:
        'udpipe' -> ('udpipe', None)
        'udpipe:yo' -> ('udpipe', 'yo')
        'spacy:en_core_web_sm' -> ('spacy', 'en_core_web_sm')
    
    Returns:
        Tuple of (backend_name, model_name or None)
    """
    if ':' in spec:
        backend, model = spec.split(':', 1)
        return backend.strip(), model.strip() if model.strip() else None
    return spec.strip(), None


def backend_supports_segmentation(backend_name: str) -> bool:
    """
    Check if a backend supports segmentation/tokenization.
    
    Backends that support segmentation:
    - flexitag: built-in segmentation
    - spacy: full pipeline with tokenization
    - stanza: supports raw text input
    - udpipe: supports raw text input
    - classla: supports raw text input
    - sentencepiece: SentencePiece-based tokenization (for unknown languages)
    
    Backends that don't support segmentation:
    - transformers: token classification only (no segmentation)
    - nametag: NER only (no segmentation)
    - udmorph: morphological tagging only (no segmentation)
    - treetagger: requires pre-tokenized input
    - flair: can work with raw text but tokenization is limited
    
    Returns:
        True if backend supports segmentation, False otherwise
    """
    backend_lower = backend_name.lower()
    
    # Backends that definitely support segmentation
    if backend_lower in ('flexitag', 'spacy', 'stanza', 'udpipe', 'classla', 'sentencepiece'):
        return True
    
    # Backends that don't support segmentation
    if backend_lower in ('transformers', 'nametag', 'udmorph', 'treetagger'):
        return False
    
    # For flair, it's ambiguous - it can work with raw text but tokenization
    # is not its primary strength. Default to False to be safe.
    if backend_lower == 'flair':
        return False
    
    # Unknown backend - assume it doesn't support segmentation
    return False


def find_segmenter_for_language(
    language: Optional[str],
    *,
    use_cache: bool = True,
    prefer_available: bool = True,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Find an available segmenter for a language.
    
    Checks backends that support segmentation in priority order:
    1. udpipe (REST, many languages)
    2. spacy (many languages)
    3. stanza (many languages)
    4. classla (specific languages)
    
    If no language-specific model is found, returns None (caller should use flexitag).
    
    Args:
        language: Language ISO code
        use_cache: Whether to use cached model catalog
        prefer_available: If True, prefer models available without download
    
    Returns:
        Tuple of (backend_name, model_name) or None if no segmenter found
    """
    if not language:
        return None
    
    from .model_catalog import get_models_for_language
    
    # Priority order for segmenter backends
    # Note: sentencepiece is not included here as it's for unknown languages
    # It should be explicitly requested or used as a fallback
    segmenter_backends = ['udpipe', 'spacy', 'stanza', 'classla']
    
    for backend_name in segmenter_backends:
        if not backend_supports_segmentation(backend_name):
            continue
        
        # Find models for this backend and language
        try:
            models = get_models_for_language(
                language,
                preferred_only=False,
                available_only=prefer_available,
                use_cache=use_cache,
            )
            backend_models = [m for m in models if m.get("backend") == backend_name]
            
            if backend_models:
                # Prefer preferred models
                preferred = [m for m in backend_models if m.get("preferred", False)]
                if preferred:
                    model_entry = preferred[0]
                else:
                    model_entry = backend_models[0]
                
                model_name = model_entry.get("model")
                if model_name:
                    return (backend_name, model_name)
        except Exception:
            # If lookup fails for this backend, try next one
            continue
    
    # No language-specific segmenter found
    return None


def get_default_segmenter_for_language(
    language: Optional[str],
    *,
    use_cache: bool = True,
    fallback_to_flexitag: bool = True,
    fallback_to_sentencepiece: bool = False,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Get default segmenter for a language.
    
    This checks:
    1. Available models for the language (udpipe, spacy, stanza, classla)
    2. Falls back to flexitag (language-independent) if no model found
    3. Optionally falls back to sentencepiece (multilingual) for unknown languages
    
    Args:
        language: Language ISO code (None for unknown languages)
        use_cache: Whether to use cached model catalog
        fallback_to_flexitag: If True, return flexitag when no language-specific model found
        fallback_to_sentencepiece: If True, prefer sentencepiece over flexitag for unknown languages
    
    Returns:
        Tuple of (backend_name, model_name) or None if no segmenter available
    """
    # First, try to find a language-specific segmenter
    if language:
        segmenter = find_segmenter_for_language(language, use_cache=use_cache)
        if segmenter:
            return segmenter
    
    # For unknown languages (no language specified), prefer sentencepiece if requested
    if not language and fallback_to_sentencepiece:
        return ('sentencepiece', None)
    
    # Fall back to flexitag (language-independent tokenization)
    if fallback_to_flexitag:
        return ('flexitag', None)
    
    return None


def apply_segmentation(
    document: Document,
    segmenter: Tuple[str, Optional[str]],
    *,
    language: Optional[str] = None,
    verbose: bool = False,
    pretokenize: bool = False,
) -> Document:
    """
    Apply pre-segmentation/tokenization to a document using the specified segmenter.
    
    Args:
        document: Input document (should have sentence.text but may not have tokens)
        segmenter: Tuple of (backend_name, model_name) or segmenter spec string
        language: Language code for segmenter (if different from document language)
        verbose: Whether to print verbose messages
        pretokenize: If True, also tokenize (not just segment). If False, only segment into sentences.
    
    Returns:
        Document with sentences (and optionally tokens) added from segmentation
    """
    # Handle both tuple and string formats
    if isinstance(segmenter, str):
        backend_name, model_name = parse_segmenter_spec(segmenter)
    else:
        backend_name, model_name = segmenter
    
    if verbose:
        model_str = f" with model '{model_name}'" if model_name else ""
        action = "pre-tokenization" if pretokenize else "pre-segmentation"
        print(f"[flexipipe] Using segmenter '{backend_name}'{model_str} for {action}", file=__import__('sys').stderr)
    
    # Create segmenter backend
    segmenter_kwargs = {}
    if model_name:
        # For different backends, model might be passed as 'model' or 'model_name'
        if backend_name in ('udpipe', 'udmorph', 'nametag'):
            segmenter_kwargs['model'] = model_name
        elif backend_name == 'sentencepiece':
            segmenter_kwargs['model_name'] = model_name
        else:
            segmenter_kwargs['model_name'] = model_name
    
    if language and backend_name != 'sentencepiece':
        # SentencePiece segmenter doesn't need language (it's language-agnostic)
        segmenter_kwargs['language'] = language
    
    try:
        # Special handling for sentencepiece segmenter
        if backend_name == 'sentencepiece':
            from .backends.sentencepiece_segmenter import SentencePieceSegmenter
            segmenter_backend = SentencePieceSegmenter(
                model_name=model_name,
                device="cpu",  # Tokenization doesn't need GPU
            )
        else:
            segmenter_backend = create_backend(
                backend_name,  # backend_type as positional argument
                **segmenter_kwargs,
            )
    except Exception as e:
        raise ValueError(
            f"Failed to create segmenter backend '{backend_name}': {e}. "
            f"Available backends for segmentation: udpipe, spacy, stanza, flexitag, classla, sentencepiece"
        ) from e
    
    # Apply segmentation using raw text mode
    # For sentencepiece segmenter, pass pretokenize flag
    if backend_name == 'sentencepiece':
        result = segmenter_backend.tag(
            document,
            use_raw_text=True,
            pretokenize=pretokenize,
        )
    else:
        # Other backends don't support pretokenize parameter
        result = segmenter_backend.tag(
            document,
            use_raw_text=True,
        )
    
    return result.document


def should_use_segmentation(
    backend_name: str,
    document: Document,
    *,
    segmenter_spec: Optional[str] = None,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    auto_detect: bool = True,
) -> bool:
    """
    Determine if pre-segmentation should be used.
    
    Note: treetagger doesn't need segmentation (it only needs tokenization),
    so we skip segmentation for treetagger to avoid clearing sentence text.
    
    Args:
        backend_name: Name of the main backend
        document: Input document
        segmenter_spec: Explicit segmenter specification (if provided, or '' to disable)
        language: Language code
        model_name: Model name (to check registry for requirements)
        auto_detect: If True, auto-detect need for segmentation
    
    Returns:
        True if segmentation should be used, False otherwise
    """
    # Treetagger doesn't need segmentation - it only needs tokenization
    # If we apply segmentation, it might clear the sentence text, preventing tokenization
    if backend_name.lower() == "treetagger":
        return False
    # If segmenter is explicitly specified (and not empty string), use it
    if segmenter_spec is not None:
        if segmenter_spec == "":
            return False  # Explicitly disabled
        return True
    
    # If auto-detection is disabled, don't use segmentation
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
                requires_segmentation = model_entry.get("requires_segmentation")
                if requires_segmentation is not None:
                    # Registry explicitly specifies whether segmentation is needed
                    return bool(requires_segmentation)
        except Exception:
            # If registry lookup fails, continue with other checks
            pass
    
    # Check if backend supports segmentation
    if backend_supports_segmentation(backend_name):
        return False  # Backend can handle it itself
    
    # Check if document already has tokens
    if any(sent.tokens for sent in document.sentences):
        return False  # Already tokenized
    
    # Check if document has text to segment
    if not any(sent.text for sent in document.sentences):
        return False  # No text to segment
    
    # Backend doesn't support segmentation and we have raw text
    # Only use segmentation if explicitly needed (don't auto-detect by default)
    return False


def _get_best_default_segmenter() -> str:
    """
    Determine the best default segmenter based on availability.
    
    Priority:
    1. sentencepiece (if transformers available)
    2. flexitag (always available)
    
    Returns:
        Segmenter name (e.g., 'sentencepiece' or 'flexitag')
    """
    # Check if sentencepiece is available (via transformers)
    try:
        import transformers
        from .backends.sentencepiece_segmenter import SentencePieceSegmenter
        return 'sentencepiece'
    except ImportError:
        pass
    
    # Fall back to flexitag (always available)
    return 'flexitag'


def resolve_segmenter_spec(
    segmenter_spec: Optional[str],
    language: Optional[str],
    *,
    backend_name: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Resolve segmenter specification to (backend, model) tuple.
    
    Args:
        segmenter_spec: Explicit segmenter spec (e.g., 'udpipe:yo' or 'udpipe' or '' to disable)
        language: Language code for auto-detection
        backend_name: Main backend name (to check registry for preferred segmenter)
        model_name: Model name (to check registry for preferred segmenter)
        use_cache: Whether to use cached model catalog
        verbose: Whether to print verbose messages
    
    Returns:
        Tuple of (backend_name, model_name) or None if no segmenter needed
    """
    # If explicit spec provided (and not empty string), parse it
    if segmenter_spec and segmenter_spec != "":
        backend, model = parse_segmenter_spec(segmenter_spec)
        if verbose:
            model_str = f":{model}" if model else ""
            print(f"[flexipipe] Using explicit segmenter: {backend}{model_str}", file=__import__('sys').stderr)
        return (backend, model)
    
    # If empty string, explicitly disabled
    if segmenter_spec == "":
        if verbose:
            print("[flexipipe] Segmentation explicitly disabled", file=__import__('sys').stderr)
        return None
    
    # Check config for default segmenter
    from .model_storage import get_default_segmenter
    default_segmenter = get_default_segmenter()
    if default_segmenter:
        backend, model = parse_segmenter_spec(default_segmenter)
        if verbose:
            model_str = f":{model}" if model else ""
            print(f"[flexipipe] Using configured default segmenter: {backend}{model_str}", file=__import__('sys').stderr)
        return (backend, model)
    
    # Check registry for preferred segmenter
    if model_name and backend_name:
        try:
            from .model_catalog import build_unified_catalog
            catalog = build_unified_catalog(use_cache=use_cache, verbose=False)
            model_key = f"{backend_name}:{model_name}"
            model_entry = catalog.get(model_key)
            if model_entry:
                preferred_segmenter = model_entry.get("preferred_segmenter")
                if preferred_segmenter:
                    # Registry specifies preferred segmenter
                    backend, model = parse_segmenter_spec(preferred_segmenter)
                    if verbose:
                        model_str = f":{model}" if model else ""
                        print(f"[flexipipe] Using registry-specified segmenter for model '{model_name}': {backend}{model_str}", file=__import__('sys').stderr)
                    return (backend, model)
        except Exception:
            # If registry lookup fails, continue with other checks
            pass
    
    # Auto-detect: try to find language-specific segmenter
    segmenter = find_segmenter_for_language(language, use_cache=use_cache)
    if segmenter:
        if verbose:
            backend, model = segmenter
            print(f"[flexipipe] Auto-detected segmenter: {backend}:{model} for language '{language}'", file=__import__('sys').stderr)
        return segmenter
    
    # No language-specific segmenter found
    # Use best default segmenter (sentencepiece if available, otherwise flexitag)
    default_segmenter = _get_best_default_segmenter()
    if verbose:
        if not language:
            print(f"[flexipipe] Unknown language, using {default_segmenter} (language-independent)", file=__import__('sys').stderr)
        else:
            print(f"[flexipipe] No language-specific segmenter found for '{language}', using {default_segmenter} (language-independent)", file=__import__('sys').stderr)
    return (default_segmenter, None)


def maybe_apply_segmentation(
    document: Document,
    backend_name: str,
    *,
    segmenter_spec: Optional[str] = None,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Tuple[Document, bool]:
    """
    Apply segmentation if needed, returning the (possibly segmented) document and whether segmentation was applied.
    
    This is the main entry point for segmentation in the pipeline. It:
    1. Determines if segmentation is needed
    2. Resolves the segmenter to use
    3. Applies segmentation if needed
    
    Args:
        document: Input document
        backend_name: Name of the main backend
        segmenter_spec: Explicit segmenter specification (e.g., 'udpipe:yo' or '' to disable)
        language: Language code
        use_cache: Whether to use cached model catalog
        verbose: Whether to print verbose messages
    
    Returns:
        Tuple of (document, segmentation_applied)
        - document: Document (possibly with sentences split from segmentation)
        - segmentation_applied: True if segmentation was applied, False otherwise
    """
    # Check if segmentation is needed
    if not should_use_segmentation(
        backend_name,
        document,
        segmenter_spec=segmenter_spec,
        language=language,
        model_name=model_name,
        auto_detect=True,
    ):
        return document, False
    
    # Resolve segmenter specification
    resolved_segmenter = resolve_segmenter_spec(
        segmenter_spec,
        language,
        backend_name=backend_name,
        model_name=model_name,
        use_cache=use_cache,
        verbose=verbose,
    )
    
    if not resolved_segmenter:
        return document, False
    
    # Apply segmentation (without tokenization - pretokenize=False for segmentation-only)
    segmented_doc = apply_segmentation(
        document,
        resolved_segmenter,
        language=language,
        verbose=verbose,
        pretokenize=False,  # Segmentation only, no tokenization
    )
    
    return segmented_doc, True

