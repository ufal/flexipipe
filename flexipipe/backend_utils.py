"""
Utility functions for backend factory functions.

This module provides common helper functions to reduce code duplication
across backend implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def validate_backend_kwargs(
    kwargs: Dict[str, Any],
    backend_name: str,
    *,
    allowed_extra: Optional[List[str]] = None,
) -> None:
    """
    Validate that no unexpected kwargs remain after processing.
    
    Args:
        kwargs: Dictionary of remaining kwargs to validate
        backend_name: Name of the backend (for error messages)
        allowed_extra: Optional list of kwargs that are allowed but not explicitly
                      handled (e.g., ["download_model", "training"])
    
    Raises:
        ValueError: If unexpected kwargs are found
    """
    if allowed_extra:
        for key in allowed_extra:
            kwargs.pop(key, None)
    
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unexpected {backend_name} backend arguments: {unexpected}")


def resolve_model_from_language(
    language: str,
    backend_name: str,
    *,
    preferred_only: bool = True,
    use_cache: bool = True,
) -> str:
    """
    Resolve a model name from a language code using the model catalog.
    
    This is a common pattern used by backends like UDPipe and Flexitag
    to automatically select a model based on language.
    
    Args:
        language: Language code (e.g., "en", "cs")
        backend_name: Name of the backend (e.g., "udpipe", "flexitag")
        preferred_only: If True, only look for preferred models first
        use_cache: Whether to use cached model catalog data
    
    Returns:
        Model name/identifier
    
    Raises:
        ValueError: If no model is found for the language
    """
    from .model_catalog import get_models_for_language
    
    # Try preferred models first if requested
    if preferred_only:
        models = get_models_for_language(language, preferred_only=True, use_cache=use_cache)
        backend_models = [m for m in models if m.get("backend") == backend_name]
        if backend_models:
            model = backend_models[0].get("model")
            if model:
                return model
    
    # Fall back to any model for the language
    all_models = get_models_for_language(language, preferred_only=False, use_cache=use_cache)
    backend_models = [m for m in all_models if m.get("backend") == backend_name]
    if backend_models:
        model = backend_models[0].get("model")
        if model:
            return model
    
    # No model found
    from .model_storage import is_running_from_teitok
    teitok_msg = "" if is_running_from_teitok() else f" Provide --model to specify a model name, or use 'python -m flexipipe info models --backend {backend_name}' to see available models."
    raise ValueError(
        f"No {backend_name} model found for language '{language}'.{teitok_msg}"
    )

