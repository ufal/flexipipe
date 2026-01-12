"""HeLI (Helsinki Language Identifier) detector using heliport package."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..language_detector_registry import LanguageDetectorSpec, register_language_detector

# Global identifier instance (lazy-loaded)
_HELIPORT_IDENTIFIER: Optional[Any] = None


def _prepare_heliport(verbose: bool) -> None:
    """Prepare heliport detector by loading the identifier."""
    global _HELIPORT_IDENTIFIER
    try:
        from heliport import Identifier
    except ImportError:
        # Try to auto-install if possible
        try:
            from ..dependency_utils import ensure_extra_installed
            ensure_extra_installed(
                "heliport",
                module_name="heliport",
                friendly_name="heliport language detector",
                allow_prompt=verbose,
            )
            from heliport import Identifier
        except (ImportError, RuntimeError):
            raise RuntimeError(
                "heliport language detector unavailable. Install with: pip install heliport "
                "or: python -m flexipipe install heliport"
            ) from None
    
    if _HELIPORT_IDENTIFIER is None:
        if verbose:
            print("[flexipipe] Initializing heliport identifier (first call may be slow)...")
        _HELIPORT_IDENTIFIER = Identifier()


def _detect_heliport(
    text: str,
    min_length: int,
    confidence_threshold: float,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """Detect language using heliport (HeLI) library."""
    if not text or len(text.strip()) < min_length:
        if verbose:
            print("[flexipipe] heliport detector skipped: input too short.")
        return None
    
    global _HELIPORT_IDENTIFIER
    
    try:
        from heliport import Identifier
    except ImportError:
        if verbose:
            print("[flexipipe] heliport not available. Install with: pip install heliport")
        return None
    
    # Initialize if needed
    if _HELIPORT_IDENTIFIER is None:
        _HELIPORT_IDENTIFIER = Identifier()
    
    try:
        # heliport.identify() returns ISO 639-3 language code (or language code with script, e.g., 'cat_latn')
        lang_code = _HELIPORT_IDENTIFIER.identify(text)
        
        if not lang_code:
            return None
        
        # Extract base language code (heliport may return codes like 'cat_latn' or 'eng_latn')
        # We'll use the part before underscore as the main language code
        base_lang_code = lang_code.split('_')[0] if '_' in lang_code else lang_code
        
        # Check if heliport supports getting confidence scores or top-N
        # Try to get more information if available
        candidates = []
        confidence = 1.0  # Default to 1.0 if not available
        
        # Check if identify_all or similar method exists
        if hasattr(_HELIPORT_IDENTIFIER, 'identify_all'):
            try:
                all_results = _HELIPORT_IDENTIFIER.identify_all(text)
                if isinstance(all_results, list) and len(all_results) > 0:
                    # Assume format is list of (lang, score) tuples or similar
                    for item in all_results[:5]:
                        if isinstance(item, tuple) and len(item) >= 2:
                            cand_lang, cand_score = item[0], item[1]
                            # Extract base language code
                            cand_base = cand_lang.split('_')[0] if '_' in cand_lang else cand_lang
                            candidates.append({
                                "language_iso": cand_base,
                                "label": cand_lang,  # Keep full code in label
                                "confidence": float(cand_score) if isinstance(cand_score, (int, float)) else 1.0,
                            })
                            if len(candidates) == 1:
                                confidence = candidates[0]["confidence"]
                        elif isinstance(item, dict):
                            # Dictionary format
                            cand_lang = item.get("language", item.get("lang", item.get("language_iso", "")))
                            cand_score = item.get("confidence", item.get("score", 1.0))
                            cand_base = cand_lang.split('_')[0] if '_' in cand_lang else cand_lang
                            candidates.append({
                                "language_iso": cand_base,
                                "label": cand_lang,
                                "confidence": float(cand_score) if isinstance(cand_score, (int, float)) else 1.0,
                            })
                            if len(candidates) == 1:
                                confidence = candidates[0]["confidence"]
            except (AttributeError, TypeError, ValueError) as e:
                if verbose:
                    print(f"[flexipipe] heliport identify_all failed: {e}")
        
        # If no candidates from identify_all, create single candidate from identify result
        if not candidates:
            candidates = [{
                "language_iso": base_lang_code,
                "label": lang_code,  # Keep full code (e.g., 'cat_latn') in label
                "confidence": confidence,
            }]
        
        # Normalize language code (heliport returns ISO 639-3, possibly with script suffix)
        from ..language_mapping import normalize_language_code
        _, iso2, iso3 = normalize_language_code(base_lang_code)
        language_iso = iso3 or iso2 or base_lang_code
        language_name = None  # heliport doesn't provide names
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            if verbose:
                print(f"[flexipipe] heliport confidence {confidence:.2%} below threshold {confidence_threshold:.2%}")
            return None
        
        result = {
            "language": lang_code,
            "language_iso": language_iso,
            "language_name": language_name,
            "confidence": confidence,
            "candidates": candidates,
        }
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"[flexipipe] heliport error: {e}")
        return None


register_language_detector(
    LanguageDetectorSpec(
        name="heliport",
        description="HeLI (Helsinki Language Identifier) via heliport package (supports 220 languages)",
        detect=_detect_heliport,
        prepare=_prepare_heliport,
        is_default=False,
    )
)

