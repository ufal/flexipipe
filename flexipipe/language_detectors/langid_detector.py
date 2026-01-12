"""langid.py language detector integration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..language_detector_registry import LanguageDetectorSpec, register_language_detector


def _prepare_langid(verbose: bool) -> None:
    """Prepare langid detector (no-op, model loads on first use)."""
    try:
        import langid  # noqa: F401
    except ImportError:
        # Try to auto-install if possible
        try:
            from ..dependency_utils import ensure_extra_installed
            ensure_extra_installed(
                "langid",
                module_name="langid",
                friendly_name="langid language detector",
                allow_prompt=verbose,
            )
            import langid  # noqa: F401
        except (ImportError, RuntimeError):
            raise RuntimeError(
                "langid language detector unavailable. Install with: pip install langid "
                "or: python -m flexipipe install langid"
            ) from None


def _detect_langid(
    text: str,
    min_length: int,
    confidence_threshold: float,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """Detect language using langid.py library."""
    if not text or len(text.strip()) < min_length:
        if verbose:
            print("[flexipipe] langid detector skipped: input too short.")
        return None
    
    try:
        import langid
    except ImportError:
        if verbose:
            print("[flexipipe] langid not available. Install with: pip install langid")
        return None
    
    try:
        # langid.classify returns (lang_code, confidence)
        # Note: langid returns log probabilities (negative values), not probabilities
        lang_code, log_confidence = langid.classify(text)
        
        # langid.rank returns list of (lang_code, log_confidence) tuples
        # Get top candidates to convert log probabilities to probabilities
        ranked = langid.rank(text)
        
        if not ranked:
            return None
        
        # Convert log probabilities to probabilities using softmax
        # langid returns log-likelihood ratios (negative values, higher/less negative is better)
        import math
        log_scores = [conf for _, conf in ranked[:10]]  # Get top 10 for normalization
        
        # Find max log score for numerical stability
        max_log = max(log_scores) if log_scores else 0.0
        
        # Convert to probabilities using softmax: exp(log_score - max_log) / sum(exp(...))
        exp_scores = [math.exp(log_score - max_log) for log_score in log_scores]
        total = sum(exp_scores)
        
        if total == 0:
            return None
        
        # Normalize to probabilities
        probabilities = [exp / total for exp in exp_scores]
        
        # Get the probability for the best language
        confidence = probabilities[0] if probabilities else 0.0
        
        # Check confidence threshold (now in probability space [0, 1])
        if confidence < confidence_threshold:
            if verbose:
                print(f"[flexipipe] langid confidence {confidence:.2%} below threshold {confidence_threshold:.2%}")
            return None
        
        # Build candidates list with probabilities
        candidates = []
        for idx, (lang, _) in enumerate(ranked[:5]):
            prob = probabilities[idx] if idx < len(probabilities) else 0.0
            candidates.append({
                "language_iso": lang,
                "label": lang,
                "confidence": prob,
            })
        
        # Normalize language code
        from ..language_mapping import normalize_language_code
        _, iso2, iso3 = normalize_language_code(lang_code)
        language_iso = iso2 or iso3 or lang_code
        language_name = None  # langid doesn't provide names
        
        result = {
            "language": lang_code,
            "language_iso": language_iso,
            "language_name": language_name,
            "confidence": confidence,  # Now in probability space [0, 1]
            "candidates": candidates,
        }
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"[flexipipe] langid error: {e}")
        return None


register_language_detector(
    LanguageDetectorSpec(
        name="langid",
        description="langid.py library (fast and accurate, supports 97 languages)",
        detect=_detect_langid,
        prepare=_prepare_langid,
        is_default=False,
    )
)

