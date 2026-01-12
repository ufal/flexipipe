"""langdetect language detector integration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..language_detector_registry import LanguageDetectorSpec, register_language_detector


def _prepare_langdetect(verbose: bool) -> None:
    """Prepare langdetect detector (no-op, but kept for consistency)."""
    try:
        import langdetect
    except ImportError:
        # Try to auto-install if possible
        try:
            from ..dependency_utils import ensure_extra_installed
            ensure_extra_installed(
                "langdetect",
                module_name="langdetect",
                friendly_name="langdetect language detector",
                allow_prompt=verbose,
            )
            import langdetect
        except (ImportError, RuntimeError):
            raise RuntimeError(
                "langdetect language detector unavailable. Install with: pip install langdetect "
                "or: python -m flexipipe install langdetect"
            ) from None


def _detect_langdetect(
    text: str,
    min_length: int,
    confidence_threshold: float,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """Detect language using langdetect library."""
    if not text or len(text.strip()) < min_length:
        if verbose:
            print("[flexipipe] langdetect detector skipped: input too short.")
        return None
    
    try:
        import langdetect
        from langdetect import detect_langs, LangDetectException
    except ImportError:
        if verbose:
            print("[flexipipe] langdetect not available. Install with: pip install langdetect")
        return None
    
    try:
        # Get all detected languages with probabilities
        detected_langs = detect_langs(text)
        
        if not detected_langs:
            return None
        
        # Best match
        best = detected_langs[0]
        lang_code = best.lang
        confidence = best.prob
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            if verbose:
                print(f"[flexipipe] langdetect confidence {confidence:.2%} below threshold {confidence_threshold:.2%}")
            return None
        
        # Build candidates list (top 5)
        candidates = []
        for lang_obj in detected_langs[:5]:
            candidates.append({
                "language_iso": lang_obj.lang,
                "label": lang_obj.lang,
                "confidence": lang_obj.prob,
            })
        
        # Normalize language code (langdetect uses ISO 639-1 codes)
        from ..language_mapping import normalize_language_code
        _, iso2, iso3 = normalize_language_code(lang_code)
        language_iso = iso2 or iso3 or lang_code
        language_name = None  # langdetect doesn't provide names
        
        result = {
            "language": lang_code,
            "language_iso": language_iso,
            "language_name": language_name,
            "confidence": confidence,
            "candidates": candidates,
        }
        
        return result
        
    except LangDetectException as e:
        if verbose:
            print(f"[flexipipe] langdetect error: {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"[flexipipe] langdetect unexpected error: {e}")
        return None


register_language_detector(
    LanguageDetectorSpec(
        name="langdetect",
        description="langdetect library (Google's language-detection port, supports 55 languages)",
        detect=_detect_langdetect,
        prepare=_prepare_langdetect,
        is_default=False,
    )
)

