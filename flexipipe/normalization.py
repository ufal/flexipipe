"""
Orthographic normalization utilities for flexipipe.

This module handles spelling correction and normalization, similar to how
flexitag uses lexicon-based normalization. It can use backends like hunspell
to correct spelling errors and set reg attributes on tokens.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .doc import Document
from .segmentation import parse_segmenter_spec


def should_use_normalization(
    backend_name: str,
    document: Document,
    *,
    normalizer_spec: Optional[str] = None,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    auto_detect: bool = True,
) -> bool:
    """
    Determine if orthographic normalization should be used.
    
    Args:
        backend_name: Name of the main backend
        document: Input document
        normalizer_spec: Explicit normalizer specification (if provided, or '' to disable)
        language: Language code
        model_name: Model name (to check registry for requirements)
        auto_detect: If True, auto-detect need for normalization
    
    Returns:
        True if normalization should be used, False otherwise
    """
    # If normalizer is explicitly specified, use it
    if normalizer_spec is not None:  # Note: empty string means "disable", None means "auto"
        if normalizer_spec == "":
            return False  # Explicitly disabled
        return True  # Explicitly specified
    
    # If auto-detection is disabled, don't use normalization
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
                requires_normalization = model_entry.get("requires_normalization")
                if requires_normalization is not None:
                    # Registry explicitly specifies whether normalization is needed
                    return bool(requires_normalization)
        except Exception:
            # If registry lookup fails, continue with other checks
            pass
    
    # Normalization is opt-in: only auto-detect if explicitly needed
    # Check if document has tokens to normalize
    if not any(sent.tokens for sent in document.sentences):
        return False  # No tokens to normalize
    
    # Don't auto-detect normalization - it's opt-in only
    # User must explicitly specify --normalizer if they want normalization
    return False


def resolve_normalizer_spec(
    normalizer_spec: Optional[str],
    language: Optional[str],
    *,
    backend_name: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Resolve normalizer specification to (backend, model) tuple.
    
    Args:
        normalizer_spec: Explicit normalizer spec (e.g., 'hunspell' or 'hunspell:en')
        language: Language code for auto-detection
        backend_name: Main backend name (for registry lookup)
        model_name: Main model name (for registry lookup)
        use_cache: Whether to use cached model catalog
        verbose: Whether to print verbose messages
    
    Returns:
        Tuple of (backend_name, model_name) or None if no normalizer needed
    """
    # If explicit spec provided (and not empty string), parse it
    if normalizer_spec and normalizer_spec != "":
        backend, model = parse_segmenter_spec(normalizer_spec)
        if verbose:
            model_str = f":{model}" if model else ""
            print(f"[flexipipe] Using explicit normalizer: {backend}{model_str}", file=__import__('sys').stderr)
        return (backend, model)
    
    # If empty string, explicitly disabled
    if normalizer_spec == "":
        if verbose:
            print("[flexipipe] Normalization explicitly disabled", file=__import__('sys').stderr)
        return None
    
    # Check config for default normalizer
    from .model_storage import get_default_normalizer
    default_normalizer = get_default_normalizer()
    if default_normalizer:
        backend, model = parse_segmenter_spec(default_normalizer)
        if verbose:
            model_str = f":{model}" if model else ""
            print(f"[flexipipe] Using configured default normalizer: {backend}{model_str}", file=__import__('sys').stderr)
        return (backend, model)
    
    # Check registry for preferred normalizer
    if model_name and backend_name:
        try:
            from .model_catalog import build_unified_catalog
            catalog = build_unified_catalog(use_cache=use_cache, verbose=False)
            model_key = f"{backend_name}:{model_name}"
            model_entry = catalog.get(model_key)
            if model_entry:
                preferred_normalizer = model_entry.get("preferred_normalizer")
                if preferred_normalizer:
                    # Registry specifies preferred normalizer
                    backend, model = parse_segmenter_spec(preferred_normalizer)
                    if verbose:
                        model_str = f":{model}" if model else ""
                        print(f"[flexipipe] Using registry-specified normalizer for model '{model_name}': {backend}{model_str}", file=__import__('sys').stderr)
                    return (backend, model)
        except Exception:
            # If registry lookup fails, continue with other checks
            pass
    
    # Auto-detect: try to find language-specific normalizer
    normalizer = find_normalizer_for_language(language, use_cache=use_cache)
    if normalizer:
        if verbose:
            backend, model = normalizer
            model_str = f":{model}" if model else ""
            print(f"[flexipipe] Auto-detected normalizer for '{language}': {backend}{model_str}", file=__import__('sys').stderr)
        return normalizer
    
    # No language-specific normalizer found - use default (hunspell)
    default_normalizer = "hunspell"
    if verbose:
        print(f"[flexipipe] No language-specific normalizer found for '{language}', using {default_normalizer} (language-independent)", file=__import__('sys').stderr)
    return (default_normalizer, None)


def find_normalizer_for_language(
    language: Optional[str],
    *,
    use_cache: bool = True,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Find an available normalizer for a language.
    
    Currently only hunspell (via phunspell package) is supported, which supports many languages.
    Checks if hunspell supports the given language.
    
    Args:
        language: Language ISO code
        use_cache: Whether to use cached model catalog
    
    Returns:
        Tuple of (backend_name, model_name) or None if no normalizer found
    """
    if not language:
        return None
    
    # Check if hunspell (phunspell package) supports this language
    try:
        import phunspell
        # phunspell supports many languages - check if dictionary exists
        # We'll try to instantiate it to see if it's available
        # Language codes in phunspell are typically ISO 639-1 (2-letter) or ISO 639-2 (3-letter)
        # Convert language code to phunspell format
        lang_code = _normalize_language_for_phunspell(language)
        if lang_code:
            # Try to check if dictionary is available (without actually loading it)
            # phunspell dictionaries are named like 'en_US', 'es_ES', etc.
            # For now, we'll assume hunspell is available if the module is installed
            # The actual dictionary loading will happen in the backend
            return ("hunspell", lang_code)
    except ImportError:
        # phunspell not installed
        pass
    except Exception:
        # Other error - continue
        pass
    
    return None


def _normalize_language_for_phunspell(language: Optional[str]) -> Optional[str]:
    """
    Convert language code to phunspell format.
    
    phunspell uses language codes like 'en_US', 'es_ES', etc.
    We'll try to map ISO codes to phunspell format.
    
    Args:
        language: Language ISO code (e.g., 'en', 'en-US', 'es', 'yo')
    
    Returns:
        phunspell language code or None
    """
    if not language:
        return None
    
    # Normalize language code
    lang_lower = language.lower().replace("-", "_")
    
    # Common mappings (phunspell uses format like 'en_US', 'es_ES')
    # For now, we'll use the language code as-is and let phunspell handle it
    # phunspell dictionaries are typically named with country codes
    # We'll try common patterns
    
    # If it's already in phunspell format (e.g., 'en_US'), return as-is
    if "_" in lang_lower and len(lang_lower.split("_")) == 2:
        return lang_lower
    
    # Map common ISO codes to phunspell format
    # This is a simplified mapping - phunspell may have more specific dictionaries
    lang_mapping = {
        "en": "en_US",
        "es": "es_ES",
        "fr": "fr_FR",
        "de": "de_DE",
        "it": "it_IT",
        "pt": "pt_PT",
        "nl": "nl_NL",
        "ru": "ru_RU",
        "pl": "pl_PL",
        "cs": "cs_CZ",
        "sk": "sk_SK",
        "sv": "sv_SE",
        "da": "da_DK",
        "no": "nb_NO",
        "fi": "fi_FI",
        "hu": "hu_HU",
        "ro": "ro_RO",
        "bg": "bg_BG",
        "hr": "hr_HR",
        "sr": "sr_RS",
        "sl": "sl_SI",
        "uk": "uk_UA",
        "be": "be_BY",
        "et": "et_EE",
        "lv": "lv_LV",
        "lt": "lt_LT",
        "el": "el_GR",
        "tr": "tr_TR",
        "ar": "ar",
        "he": "he_IL",
        "fa": "fa_IR",
        "hi": "hi_IN",
        "th": "th_TH",
        "vi": "vi_VN",
        "ko": "ko_KR",
        "ja": "ja_JP",
        "zh": "zh_CN",
    }
    
    # Check if we have a direct mapping
    if lang_lower in lang_mapping:
        return lang_mapping[lang_lower]
    
    # If it's a 2-letter code, try to construct phunspell format
    # (use language code + uppercase version as country code)
    if len(lang_lower) == 2:
        return f"{lang_lower}_{lang_lower.upper()}"
    
    # If it's a 3-letter code, try to find a 2-letter equivalent
    # For now, just return the language code and let phunspell handle it
    return lang_lower


def apply_normalization(
    document: Document,
    normalizer: Tuple[str, Optional[str]],
    *,
    language: Optional[str] = None,
    verbose: bool = False,
) -> Document:
    """
    Apply orthographic normalization to a document using the specified normalizer.
    
    Args:
        document: Input document (should have tokens)
        normalizer: Tuple of (backend_name, model_name) or normalizer spec string
        language: Language code for normalizer (if different from document language)
        verbose: Whether to print verbose messages
    
    Returns:
        Document with reg attributes set on tokens
    """
    # Handle both tuple and string formats
    if isinstance(normalizer, str):
        backend_name, model_name = parse_segmenter_spec(normalizer)
    else:
        backend_name, model_name = normalizer
    
    if verbose:
        model_str = f" with model '{model_name}'" if model_name else ""
        print(f"[flexipipe] Using normalizer '{backend_name}'{model_str} for orthographic normalization", file=__import__('sys').stderr)
    
    # Create normalizer backend
    from .backend_registry import create_backend
    
    # Normalizers are special backends that only do normalization
    # They should implement a normalize() method
    try:
        # Use the language or model_name to determine the dictionary
        normalizer_lang = language or model_name or "en_US"
        normalizer_backend = create_backend(backend_name, model=normalizer_lang, language=normalizer_lang)
    except Exception as e:
        raise ValueError(
            f"Failed to create normalizer backend '{backend_name}': {e}. "
            f"Available normalizers: hunspell"
        ) from e
    
    # Check if backend supports normalization
    if not hasattr(normalizer_backend, 'normalize'):
        raise ValueError(
            f"Backend '{backend_name}' does not support normalization. "
            f"Available normalizers: hunspell"
        )
    
    # Apply normalization to each sentence
    # Only set reg attribute, don't modify form (like flexitag does)
    # After normalization, we have tokenized text that should be sent to backend in pre-tokenized mode
    
    tokens_normalized = False
    # Create normalization map: (sentence_idx, normalized_form) -> (original_form, reg_value)
    # This allows us to restore original forms after backend processing
    normalization_map = {}
    
    for sent_idx, sentence in enumerate(document.sentences):
        for token in sentence.tokens:
            if token.form:
                # Skip normalization for punctuation-only tokens (periods, commas, etc.)
                # Punctuation shouldn't be normalized
                import unicodedata
                is_punctuation = all(unicodedata.category(ch).startswith("P") for ch in token.form if ch)
                if is_punctuation:
                    continue
                
                # Normalize the token form
                normalized = normalizer_backend.normalize(token.form, language=language or model_name)
                # Only apply normalization if we got a different form back
                # (normalize() returns None if word is correct)
                if normalized and normalized != token.form:
                    # Set reg attribute with normalized form (don't modify form)
                    # Allow case-only changes (e.g., "iz" -> "Iz") as these are valid normalizations
                    if not token.reg or token.reg == "_":
                        token.reg = normalized
                    tokens_normalized = True
                    # Store mapping: when backend returns normalized form, we can restore original
                    # Key is (sentence_idx, normalized_form), value is (original_form, reg_value)
                    normalization_map[(sent_idx, normalized)] = (token.form, normalized)
    
    # Mark that normalization was applied (for forcing pre-tokenized mode)
    if tokens_normalized:
        document.meta["_normalization_applied"] = True
        document.meta["_normalization_map"] = normalization_map
    
    return document


def maybe_apply_normalization(
    document: Document,
    backend_name: str,
    *,
    normalizer_spec: Optional[str] = None,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Tuple[Document, bool]:
    """
    Apply orthographic normalization if needed.
    
    This is the main entry point for normalization. It determines if normalization
    is needed and applies it using the appropriate normalizer.
    
    Args:
        document: Input document
        backend_name: Name of the main backend
        normalizer_spec: Explicit normalizer specification (e.g., 'hunspell' or 'hunspell:en')
        language: Language code
        model_name: Model name (for registry lookup)
        use_cache: Whether to use cached model catalog
        verbose: Whether to print verbose messages
    
    Returns:
        Tuple of (document, normalization_applied)
    """
    # Check if normalization is needed
    if not should_use_normalization(
        backend_name,
        document,
        normalizer_spec=normalizer_spec,
        language=language,
        model_name=model_name,
        auto_detect=True,
    ):
        return document, False
    
    # Resolve normalizer specification
    resolved_normalizer = resolve_normalizer_spec(
        normalizer_spec,
        language,
        backend_name=backend_name,
        model_name=model_name,
        use_cache=use_cache,
        verbose=verbose,
    )
    
    if not resolved_normalizer:
        return document, False
    
    # Apply normalization
    normalized_doc = apply_normalization(
        document,
        resolved_normalizer,
        language=language,
        verbose=verbose,
    )
    
    return normalized_doc, True

