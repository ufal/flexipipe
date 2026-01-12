"""
Hunspell normalizer backend for flexipipe.

This backend uses phunspell (a pure Python Hunspell port) to perform
orthographic normalization and spelling correction.
"""

from __future__ import annotations

import sys
from typing import Optional

from ..backend_spec import BackendSpec
from ..dependency_utils import ensure_extra_installed


def _create_hunspell_normalizer(
    model: Optional[str] = None,
    language: Optional[str] = None,
    **kwargs,
):
    """Create a Hunspell normalizer backend (using phunspell package)."""
    ensure_extra_installed(
        "phunspell",
        module_name="phunspell",
        friendly_name="Hunspell normalizer",
    )
    
    import phunspell
    
    # Determine language code
    lang_code = model or language
    if not lang_code:
        # Default to English if no language specified
        lang_code = "en_US"
    
    # Normalize language code for phunspell
    # phunspell uses format like 'en_US', 'es_ES', etc.
    lang_code = _normalize_language_code(lang_code)
    
    # Create phunspell instance (phunspell is the Python package, Hunspell is the technology)
    try:
        pspell = phunspell.Phunspell(lang_code)
    except Exception as e:
        # If dictionary not found, try without country code
        if "_" in lang_code:
            lang_base = lang_code.split("_")[0]
            try:
                pspell = phunspell.Phunspell(lang_base)
                lang_code = lang_base
            except Exception:
                raise ValueError(
                    f"Hunspell dictionary not found for language '{lang_code}' or '{lang_base}'. "
                    f"Available dictionaries can be checked with: python -c 'import phunspell; print(phunspell.Phunspell.list_dicts())'"
                ) from e
        else:
            raise ValueError(
                f"Hunspell dictionary not found for language '{lang_code}'. "
                f"Available dictionaries can be checked with: python -c 'import phunspell; print(phunspell.Phunspell.list_dicts())'"
            ) from e
    
    return HunspellNormalizer(pspell, lang_code)


def _normalize_language_code(lang_code: str) -> str:
    """
    Normalize language code to phunspell format.
    
    phunspell uses format like 'en_US', 'es_ES', etc.
    """
    if not lang_code:
        return "en_US"
    
    lang_lower = lang_code.lower().replace("-", "_")
    
    # If already in phunspell format, return as-is
    if "_" in lang_lower and len(lang_lower.split("_")) == 2:
        parts = lang_lower.split("_")
        return f"{parts[0]}_{parts[1].upper()}"
    
    # Map common ISO codes to phunspell format
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
    if len(lang_lower) == 2:
        return f"{lang_lower}_{lang_lower.upper()}"
    
    # Return as-is and let phunspell handle it
    return lang_lower


class HunspellNormalizer:
    """
    Hunspell-based orthographic normalizer.
    
    This normalizer uses phunspell (a Python port of Hunspell) to check spelling and suggest corrections.
    It sets the 'reg' attribute on tokens with the corrected spelling.
    """
    
    def __init__(self, pspell, lang_code: str):
        """
        Initialize Hunspell normalizer.
        
        Args:
            pspell: phunspell.Phunspell instance
            lang_code: Language code used
        """
        self.pspell = pspell
        self.lang_code = lang_code
    
    def normalize(self, word: str, language: Optional[str] = None) -> Optional[str]:
        """
        Normalize a word using Hunspell (via phunspell).
        
        If the word is spelled correctly, returns None (no normalization needed).
        If the word is misspelled, returns the first suggestion (corrected spelling).
        
        Args:
            word: Word to normalize
            language: Language code (ignored, uses the language from initialization)
        
        Returns:
            Normalized form if correction found, None if word is correct or no correction available
        """
        if not word:
            return None
        
        # Check if word is spelled correctly
        if self.pspell.lookup(word):
            # Word is correct - no normalization needed
            return None
        
        # Word is misspelled - get suggestions
        suggestions = self.pspell.suggest(word)
        # suggestions is a generator, so convert to list or get first item
        try:
            first_suggestion = next(suggestions)
            return first_suggestion
        except StopIteration:
            # No suggestions available
            pass
        
        # No suggestions available - word might be unknown but not necessarily wrong
        # (e.g., proper nouns, technical terms)
        return None
    
    def check(self, word: str) -> bool:
        """
        Check if a word is spelled correctly.
        
        Args:
            word: Word to check
        
        Returns:
            True if word is spelled correctly, False otherwise
        """
        if not word:
            return True
        return self.pspell.lookup(word)
    
    def suggest(self, word: str, max_suggestions: int = 10) -> list[str]:
        """
        Get spelling suggestions for a word.
        
        Args:
            word: Word to get suggestions for
            max_suggestions: Maximum number of suggestions to return
        
        Returns:
            List of suggested corrections
        """
        if not word:
            return []
        # suggestions is a generator, so convert to list and slice
        suggestions = list(self.pspell.suggest(word))
        return suggestions[:max_suggestions]


# Backend specification for auto-discovery
BACKEND_SPEC = BackendSpec(
    name="hunspell",
    factory=_create_hunspell_normalizer,
    description="Hunspell-based orthographic normalizer (spelling correction using Hunspell dictionaries via phunspell)",
    install_instructions="Install via: flexipipe install hunspell (installs phunspell package)",
)

