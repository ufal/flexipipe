"""
Comprehensive language mapping table.

This module provides a hard-coded, pre-compiled mapping of language identifiers
to standardized language metadata. This includes:
- ISO 639-1 (2-letter codes)
- ISO 639-2 (3-letter codes)
- ISO 639-3 (3-letter codes, more comprehensive)
- Language names (various capitalizations and variants)
- Common variants and aliases

This mapping is compiled once and used for fast language matching and normalization.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# Comprehensive language mapping
# Format: (iso_639_1, iso_639_2, iso_639_3, primary_name, [variants])
_LANGUAGE_MAPPINGS: List[Tuple[str, str, str, str, List[str]]] = [
    # Format: (iso_639_1, iso_639_2, iso_639_3, primary_name, [variants])
    # Common languages with multiple identifiers
    ("af", "afr", "afr", "Afrikaans", ["afrikaans"]),
    ("ar", "ara", "ara", "Arabic", ["arabic", "العربية"]),
    ("bg", "bul", "bul", "Bulgarian", ["bulgarian", "български"]),
    ("ca", "cat", "cat", "Catalan", ["catalan", "català"]),
    ("cs", "ces", "ces", "Czech", ["czech", "čeština"]),
    ("da", "dan", "dan", "Danish", ["danish", "dansk"]),
    ("de", "deu", "deu", "German", ["german", "deutsch"]),
    ("el", "ell", "ell", "Greek", ["greek", "ελληνικά"]),
    ("en", "eng", "eng", "English", ["english"]),
    ("es", "spa", "spa", "Spanish", ["spanish", "español", "espanol"]),
    ("et", "est", "est", "Estonian", ["estonian", "eesti"]),
    ("eu", "eus", "eus", "Basque", ["basque", "euskara"]),
    ("fa", "fas", "fas", "Persian", ["persian", "farsi", "فارسی"]),
    ("fi", "fin", "fin", "Finnish", ["finnish", "suomi"]),
    ("fr", "fra", "fra", "French", ["french", "français", "francais"]),
    ("ga", "gle", "gle", "Irish", ["irish", "gaeilge"]),
    ("gl", "glg", "glg", "Galician", ["galician", "galego"]),
    ("he", "heb", "heb", "Hebrew", ["hebrew", "עברית"]),
    ("hi", "hin", "hin", "Hindi", ["hindi", "हिन्दी"]),
    ("hr", "hrv", "hrv", "Croatian", ["croatian", "hrvatski"]),
    ("hu", "hun", "hun", "Hungarian", ["hungarian", "magyar"]),
    ("id", "ind", "ind", "Indonesian", ["indonesian", "bahasa indonesia"]),
    ("it", "ita", "ita", "Italian", ["italian", "italiano"]),
    ("ja", "jpn", "jpn", "Japanese", ["japanese", "日本語"]),
    ("ko", "kor", "kor", "Korean", ["korean", "한국어"]),
    ("la", "lat", "lat", "Latin", ["latin", "latina"]),
    ("lv", "lav", "lav", "Latvian", ["latvian", "latviešu"]),
    ("nl", "nld", "nld", "Dutch", ["dutch", "nederlands"]),
    ("no", "nor", "nor", "Norwegian", ["norwegian", "norsk"]),
    ("pl", "pol", "pol", "Polish", ["polish", "polski"]),
    ("pt", "por", "por", "Portuguese", ["portuguese", "português", "portugues"]),
    ("ro", "ron", "ron", "Romanian", ["romanian", "română", "romana"]),
    ("ru", "rus", "rus", "Russian", ["russian", "русский"]),
    ("sk", "slk", "slk", "Slovak", ["slovak", "slovenčina"]),
    ("sl", "slv", "slv", "Slovenian", ["slovenian", "slovenski"]),
    ("sv", "swe", "swe", "Swedish", ["swedish", "svenska"]),
    ("tr", "tur", "tur", "Turkish", ["turkish", "türkçe", "turkce"]),
    ("uk", "ukr", "ukr", "Ukrainian", ["ukrainian", "українська"]),
    ("vi", "vie", "vie", "Vietnamese", ["vietnamese", "tiếng việt", "tieng viet"]),
    ("zh", "zho", "zho", "Chinese", ["chinese", "中文", "mandarin"]),
    # Additional languages
    ("is", "isl", "isl", "Icelandic", ["icelandic", "íslenska"]),
    ("got", "got", "got", "Gothic", ["gothic"]),
    ("sq", "sqi", "sqi", "Albanian", ["albanian", "shqip"]),
    ("hy", "hye", "hye", "Armenian", ["armenian", "հայերեն"]),
    ("be", "bel", "bel", "Belarusian", ["belarusian", "беларуская"]),
    ("bor", None, "bor", "Bororo", ["bororo"]),  # ISO 639-3 only
    ("xcl", None, "xcl", "Classical Armenian", ["classical_armenian", "classical armenian"]),  # ISO 639-3 only
    ("lzh", None, "lzh", "Classical Chinese", ["classical_chinese", "classical chinese", "文言"]),  # ISO 639-3 only
    ("cop", "cop", "cop", "Coptic", ["coptic"]),
    ("fo", "fao", "fao", "Faroese", ["faroese", "føroyskt"]),
    ("ka", "kat", "kat", "Georgian", ["georgian", "ქართული"]),
    ("hsb", None, "hsb", "Upper Sorbian", ["upper sorbian", "sorbian (upper)"]),  # ISO 639-3 only
    # Add more languages as needed
]

# Build lookup dictionaries for fast access
_LANGUAGE_BY_CODE: Dict[str, Dict[str, str]] = {}
_LANGUAGE_BY_NAME: Dict[str, Dict[str, str]] = {}


def _build_language_mappings() -> None:
    """Build lookup dictionaries from language mappings."""
    global _LANGUAGE_BY_CODE, _LANGUAGE_BY_NAME
    
    if _LANGUAGE_BY_CODE:  # Already built
        return
    
    for iso_1, iso_2, iso_3, primary_name, variants in _LANGUAGE_MAPPINGS:
        lang_data = {
            "iso_639_1": iso_1,
            "iso_639_2": iso_2,
            "iso_639_3": iso_3,
            "primary_name": primary_name,
        }
        
        # Index by all codes (skip None values)
        if iso_1:
            _LANGUAGE_BY_CODE[iso_1.lower()] = lang_data
        if iso_2:
            _LANGUAGE_BY_CODE[iso_2.lower()] = lang_data
        if iso_3:
            _LANGUAGE_BY_CODE[iso_3.lower()] = lang_data
        
        # Index by primary name (various capitalizations)
        _LANGUAGE_BY_NAME[primary_name.lower()] = lang_data
        _LANGUAGE_BY_NAME[primary_name.title()] = lang_data
        _LANGUAGE_BY_NAME[primary_name.upper()] = lang_data
        
        # Index by variants
        for variant in variants:
            _LANGUAGE_BY_NAME[variant.lower()] = lang_data
            _LANGUAGE_BY_NAME[variant.title()] = lang_data
            _LANGUAGE_BY_NAME[variant.upper()] = lang_data


def normalize_language_code(language: str) -> Tuple[str, str, str]:
    """
    Normalize a language identifier to standardized ISO codes.
    
    Args:
        language: Language identifier (code or name)
        
    Returns:
        Tuple of (iso_639_1, iso_639_2, iso_639_3) or (None, None, None) if not found
    """
    _build_language_mappings()
    
    lang_lower = language.lower().strip()
    
    # Try code lookup first
    if lang_lower in _LANGUAGE_BY_CODE:
        lang_data = _LANGUAGE_BY_CODE[lang_lower]
        return (lang_data["iso_639_1"], lang_data["iso_639_2"], lang_data["iso_639_3"])
    
    # Try name lookup
    if lang_lower in _LANGUAGE_BY_NAME:
        lang_data = _LANGUAGE_BY_NAME[lang_lower]
        return (lang_data["iso_639_1"], lang_data["iso_639_2"], lang_data["iso_639_3"])
    
    # Try partial matches (e.g., "spanish" matches "Spanish")
    for key, lang_data in _LANGUAGE_BY_NAME.items():
        if lang_lower in key or key in lang_lower:
            return (lang_data["iso_639_1"], lang_data["iso_639_2"], lang_data["iso_639_3"])
    
    return (None, None, None)


def get_language_metadata(language: str) -> Dict[str, str]:
    """
    Get comprehensive language metadata for a language identifier.
    
    Args:
        language: Language identifier (code or name)
        
    Returns:
        Dictionary with iso_639_1, iso_639_2, iso_639_3, primary_name, and variants
    """
    _build_language_mappings()
    
    lang_lower = language.lower().strip()
    
    # Try code lookup first
    if lang_lower in _LANGUAGE_BY_CODE:
        return _LANGUAGE_BY_CODE[lang_lower].copy()
    
    # Try name lookup
    if lang_lower in _LANGUAGE_BY_NAME:
        return _LANGUAGE_BY_NAME[lang_lower].copy()
    
    # Try partial matches
    for key, lang_data in _LANGUAGE_BY_NAME.items():
        if lang_lower in key or key in lang_lower:
            return lang_data.copy()
    
    # Return None values if not found
    return {
        "iso_639_1": None,
        "iso_639_2": None,
        "iso_639_3": None,
        "primary_name": None,
    }

