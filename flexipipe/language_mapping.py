"""
Comprehensive language mapping table.

This module provides a mapping of language identifiers to standardized language metadata.
It loads from a JSON file in flexipipe-models/resources/languages.json (if available),
with a fallback to a hard-coded list. This allows manual corrections for edge cases.

The mapping includes:
- ISO 639-1 (2-letter codes)
- ISO 639-2 (3-letter codes)
- ISO 639-3 (3-letter codes, more comprehensive)
- Language names (various capitalizations and variants)
- Common variants and aliases
- Non-standard codes (e.g., "old_ch" for Old Church Slavonic)

This mapping is compiled once and used for fast language matching and normalization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import requests
except ImportError:
    requests = None

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
    ("my", "mya", "mya", "Burmese", ["burmese", "myanmar"]),
    ("lv", "lav", "lav", "Latvian", ["latvian", "latviešu"]),
    ("nl", "nld", "nld", "Dutch", ["dutch", "nederlands"]),
    ("no", "nor", "nor", "Norwegian", ["norwegian", "norsk"]),
    ("pl", "pol", "pol", "Polish", ["polish", "polski"]),
    ("pt", "por", "por", "Portuguese", ["portuguese", "português", "portugues"]),
    ("ro", "ron", "ron", "Romanian", ["romanian", "română", "romana"]),
    ("ru", "rus", "rus", "Russian", ["russian", "русский"]),
    ("sk", "slk", "slk", "Slovak", ["slovak", "slovenčina"]),
    ("sl", "slv", "slv", "Slovenian", ["slovenian", "slovenski"]),
    ("sw", "swa", "swa", "Swahili", ["swahili", "kiswahili"]),
    ("sv", "swe", "swe", "Swedish", ["swedish", "svenska"]),
    ("tr", "tur", "tur", "Turkish", ["turkish", "türkçe", "turkce"]),
    ("uk", "ukr", "ukr", "Ukrainian", ["ukrainian", "українська"]),
    ("vi", "vie", "vie", "Vietnamese", ["vietnamese", "tiếng việt", "tieng viet"]),
    ("zh", "zho", "zho", "Chinese", ["chinese", "中文", "mandarin"]),
    ("ug", "uig", "uig", "Uyghur", ["uyghur", "uygur", "uighur", "ئۇيغۇرچە"]),
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
    # Historical and non-standard languages
    ("cu", "chu", "chu", "Church Slavonic", ["church slavonic", "old church slavonic", "old_church_slavonic", "old_ch"]),
    ("orv", None, "orv", "Old East Slavic", ["old east slavic", "old_east_slavic", "old russian", "ruthenian"]),
    # Add more languages as needed
]

# Default remote language mapping URL
DEFAULT_LANGUAGE_MAPPING_URL = "https://raw.githubusercontent.com/ufal/flexipipe-models/main/resources/languages.json"

# Build lookup dictionaries for fast access
_LANGUAGE_BY_CODE: Dict[str, Dict[str, str]] = {}
_LANGUAGE_BY_NAME: Dict[str, Dict[str, str]] = {}
_LANGUAGE_MAPPINGS_LOADED = False

# Optional axis + parsing metadata loaded from languages.json
_LANGUAGE_AXES: Dict[str, Any] = {}
_LANGUAGE_PARSING_HEURISTICS: Dict[str, Any] = {}


def _set_language_axes_metadata(axes: Dict[str, Any], parsing_heuristics: Dict[str, Any]) -> None:
    """
    Store axis semantics and parsing heuristics loaded from languages.json.

    This keeps the responsibility for loading languages.json in this module,
    while allowing other parts of the codebase to query the axis configuration
    without needing to know about the remote source or caching.
    """
    global _LANGUAGE_AXES, _LANGUAGE_PARSING_HEURISTICS
    if isinstance(axes, dict):
        _LANGUAGE_AXES = axes
    else:
        _LANGUAGE_AXES = {}
    if isinstance(parsing_heuristics, dict):
        _LANGUAGE_PARSING_HEURISTICS = parsing_heuristics
    else:
        _LANGUAGE_PARSING_HEURISTICS = {}


def get_language_axes() -> Dict[str, Any]:
    """
    Return the axis semantics loaded from languages.json, or an empty dict.

    The structure follows the `axes` key in the flexipipe-models languages.json.
    """
    return _LANGUAGE_AXES


def get_language_parsing_heuristics() -> Dict[str, Any]:
    """
    Return the parsing heuristics loaded from languages.json, or an empty dict.

    The structure follows the `parsing_heuristics` key in languages.json.
    """
    return _LANGUAGE_PARSING_HEURISTICS


def _load_language_mappings_from_json(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    verbose: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Load language mappings (and optionally axis metadata) from a JSON file.
    
    The JSON file is expected to be the flexipipe-models `languages.json`, with
    at least the following structure:
    
        {
            "languages": [
                {
                    "iso_639_1": "en",
                    "iso_639_2": "eng",
                    "iso_639_3": "eng",
                    "primary_name": "English",
                    "variants": ["english"],
                    "aliases": ["en", "eng"]
                },
                ...
            ],
            "axes": {
                ...
            },
            "parsing_heuristics": {
                ...
            }
        }
    
    Historically this function returned only a flat list of language mapping
    dictionaries. For backwards compatibility we still return such a list, but
    when the JSON has a top-level dict we will also extract and cache the
    optional `axes` and `parsing_heuristics` sections for other parts of the
    code to consume.
    
    Args:
        url: URL to the JSON file. If None, uses the default from flexipipe-models.
        use_cache: Whether to use cached version if available.
        verbose: Whether to print debug messages.
        
    Returns:
        List of language mapping dictionaries, or None if loading fails.
    """
    from .model_storage import get_cache_dir, read_model_cache_entry, write_model_cache_entry
    
    mapping_url = url or DEFAULT_LANGUAGE_MAPPING_URL
    
    # Try to load from cache first
    if use_cache:
        cache_key = "language_mappings"
        cache_dir = get_cache_dir()
        if cache_dir:
            cached = read_model_cache_entry(cache_key, max_age_seconds=86400)  # 24 hours
            # Old cache format: list of language dicts
            if isinstance(cached, list):
                if verbose:
                    print(
                        f"[flexipipe] Loaded {len(cached)} language mappings from cache",
                        file=__import__("sys").stderr,
                    )
                return cached
            # New cache format: dict with "languages" and optional "axes"/"parsing_heuristics"
            if isinstance(cached, dict) and "languages" in cached:
                languages = cached.get("languages") or []
                if verbose:
                    print(
                        f"[flexipipe] Loaded {len(languages)} language mappings from cache (with metadata)",
                        file=__import__("sys").stderr,
                    )
                # Cache axes and parsing heuristics for global access
                _set_language_axes_metadata(
                    cached.get("axes") or {},
                    cached.get("parsing_heuristics") or {},
                )
                if isinstance(languages, list):
                    return languages
    
    # Try to fetch from remote URL
    if mapping_url.startswith("http://") or mapping_url.startswith("https://"):
        if not requests:
            if verbose:
                print("[flexipipe] Warning: 'requests' not available, cannot fetch language mappings from remote", file=sys.stderr)
            return None
        
        try:
            response = requests.get(mapping_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and "languages" in data:
                languages = data.get("languages") or []
                # Extract and store axes + parsing heuristics (may be empty)
                axes = data.get("axes") or {}
                parsing = data.get("parsing_heuristics") or {}
                _set_language_axes_metadata(axes, parsing)
                cache_payload: Any = {"languages": languages, "axes": axes, "parsing_heuristics": parsing}
            elif isinstance(data, list):
                languages = data
                cache_payload = languages
            else:
                if verbose:
                    print(f"[flexipipe] Warning: Invalid language mapping JSON format", file=sys.stderr)
                return None
            
            # Cache the result
            if use_cache and cache_dir:
                try:
                    write_model_cache_entry(cache_key, cache_payload)
                except (OSError, PermissionError):
                    pass
            
            if verbose:
                print(
                    f"[flexipipe] Loaded {len(languages)} language mappings from {mapping_url}",
                    file=sys.stderr,
                )
            return languages
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: Failed to load language mappings from {mapping_url}: {exc}", file=sys.stderr)
            return None
    
    # Try to load from local file
    elif mapping_url.startswith("file://"):
        file_path = Path(mapping_url[7:])
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "languages" in data:
                languages = data["languages"]
            elif isinstance(data, list):
                languages = data
            else:
                return None
            
            if verbose:
                print(f"[flexipipe] Loaded {len(languages)} language mappings from {file_path}", file=sys.stderr)
            return languages
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: Failed to load language mappings from {file_path}: {exc}", file=sys.stderr)
            return None
    
    return None


def _build_language_mappings(
    *,
    force_reload: bool = False,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> None:
    """Build lookup dictionaries from language mappings."""
    global _LANGUAGE_BY_CODE, _LANGUAGE_BY_NAME, _LANGUAGE_MAPPINGS_LOADED
    
    if _LANGUAGE_MAPPINGS_LOADED and not force_reload:
        return
    
    # Reset when forcing reload
    if force_reload:
        _LANGUAGE_BY_CODE = {}
        _LANGUAGE_BY_NAME = {}
        _LANGUAGE_MAPPINGS_LOADED = False
    
    # Try to load from JSON file first (allows manual corrections)
    json_mappings = _load_language_mappings_from_json(
        use_cache=not refresh_cache,
        verbose=verbose,
    )
    
    # Start with hard-coded mappings as base
    mappings_to_process: List[Tuple[Optional[str], Optional[str], Optional[str], str, List[str]]] = list(_LANGUAGE_MAPPINGS)
    
    # Process JSON mappings and add/override hard-coded ones
    # JSON takes precedence for corrections
    if json_mappings:
        # Create a set of existing codes to check for duplicates
        existing_codes = set()
        for iso_1, iso_2, iso_3, _, _ in mappings_to_process:
            if iso_1:
                existing_codes.add(iso_1.lower())
            if iso_2:
                existing_codes.add(iso_2.lower())
            if iso_3:
                existing_codes.add(iso_3.lower())
        
        # Process JSON mappings
        for lang_entry in json_mappings:
            if not isinstance(lang_entry, dict):
                continue
            
            iso_1 = lang_entry.get("iso_639_1")
            iso_2 = lang_entry.get("iso_639_2")
            iso_3 = lang_entry.get("iso_639_3")
            primary_name = lang_entry.get("primary_name", "")
            variants = lang_entry.get("variants", [])
            aliases = lang_entry.get("aliases", [])
            
            # Also collect dialect-specific aliases (e.g., oci-ara from dialects.ara.aliases)
            dialects = lang_entry.get("dialects", {})
            if isinstance(dialects, dict):
                for dialect_code, dialect_info in dialects.items():
                    if isinstance(dialect_info, dict):
                        dialect_aliases = dialect_info.get("aliases", [])
                        if dialect_aliases:
                            aliases.extend(dialect_aliases)
            
            # Combine variants and aliases
            all_variants = list(variants) + list(aliases)
            
            # Convert None to empty string for consistency with hard-coded format
            iso_1 = iso_1 if iso_1 else None
            iso_2 = iso_2 if iso_2 else None
            iso_3 = iso_3 if iso_3 else None
            
            # Check if this language already exists (by any of its codes)
            # If so, replace the existing entry; otherwise, add it
            lang_codes = [c.lower() for c in [iso_1, iso_2, iso_3] if c]
            found_existing = False
            for i, (existing_iso_1, existing_iso_2, existing_iso_3, _, _) in enumerate(mappings_to_process):
                existing_codes_for_entry = [c.lower() for c in [existing_iso_1, existing_iso_2, existing_iso_3] if c]
                if any(code in existing_codes_for_entry for code in lang_codes if code):
                    # Replace existing entry with JSON version
                    mappings_to_process[i] = (iso_1, iso_2, iso_3, primary_name, all_variants)
                    found_existing = True
                    break
            
            if not found_existing:
                # Add new entry
                mappings_to_process.append((iso_1, iso_2, iso_3, primary_name, all_variants))
    
    # Build a mapping from language codes to their aliases (including dialect aliases)
    # This allows us to index aliases to point to the correct base language
    # We'll match by any ISO code (1, 2, or 3) to handle variations
    code_to_aliases: Dict[str, List[str]] = {}
    if json_mappings:
        for lang_entry in json_mappings:
            if not isinstance(lang_entry, dict):
                continue
            entry_iso_1 = lang_entry.get("iso_639_1")
            entry_iso_2 = lang_entry.get("iso_639_2")
            entry_iso_3 = lang_entry.get("iso_639_3")
            entry_aliases = list(lang_entry.get("aliases", []))
            # Also collect dialect-specific aliases (e.g., oci-ara from dialects.ara.aliases)
            entry_dialects = lang_entry.get("dialects", {})
            if isinstance(entry_dialects, dict):
                for dialect_info in entry_dialects.values():
                    if isinstance(dialect_info, dict):
                        entry_aliases.extend(dialect_info.get("aliases", []))
            # Index by all ISO codes that exist (1, 2, or 3)
            for iso_code in [entry_iso_1, entry_iso_2, entry_iso_3]:
                if iso_code:
                    code_to_aliases[iso_code.lower()] = entry_aliases
    
    # Build lookup dictionaries
    for iso_1, iso_2, iso_3, primary_name, variants in mappings_to_process:
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
        if primary_name:
            _LANGUAGE_BY_NAME[primary_name.lower()] = lang_data
            _LANGUAGE_BY_NAME[primary_name.title()] = lang_data
            _LANGUAGE_BY_NAME[primary_name.upper()] = lang_data
        
        # Index by variants
        for variant in variants:
            if variant:
                _LANGUAGE_BY_NAME[variant.lower()] = lang_data
                _LANGUAGE_BY_NAME[variant.title()] = lang_data
                _LANGUAGE_BY_NAME[variant.upper()] = lang_data
        
        # Index by aliases (including dialect-specific aliases like oci-ara)
        # Match this language's codes to find its aliases (try any ISO code)
        for iso_code in [iso_1, iso_2, iso_3]:
            if iso_code and iso_code.lower() in code_to_aliases:
                for alias in code_to_aliases[iso_code.lower()]:
                    if alias:
                        _LANGUAGE_BY_CODE[alias.lower()] = lang_data
                break  # Found aliases, no need to check other codes
    
    _LANGUAGE_MAPPINGS_LOADED = True


def reload_language_mappings(*, refresh_cache: bool = False, verbose: bool = False) -> None:
    """Force reload of language mappings from remote/cache."""
    _build_language_mappings(force_reload=True, refresh_cache=refresh_cache, verbose=verbose)
    _reset_language_base()


# ---------------------------------------------------------------------------
# Generated base: all of Glottolog and ISO 639-3 (flexipipe-models
# resources/languages-base.json, made by tools/build_languages_base.py).
# It is consulted only when the hand table above has no exact match, and it
# is loaded lazily: languages the hand table knows never trigger it. Hand
# entries always win.
# ---------------------------------------------------------------------------

DEFAULT_LANGUAGE_BASE_URL = "https://raw.githubusercontent.com/ufal/flexipipe-models/main/resources/languages-base.json"
LANGUAGE_BASE_ENV = "FLEXIPIPE_LANGUAGE_BASE"  # a local path or URL overriding the default

_BASE_LOADED = False
_BASE_BY_CODE: Dict[str, Dict[str, Any]] = {}   # iso 639-1/2b/2t/3 and glottocode, lowercased
_BASE_BY_NAME: Dict[str, Dict[str, Any]] = {}   # unique names of languages, lowercased
_BASE_BY_GLOTTOCODE: Dict[str, Dict[str, Any]] = {}
_BASE_RETIRED: Dict[str, Dict[str, Any]] = {}


def _reset_language_base() -> None:
    global _BASE_LOADED
    _BASE_LOADED = False
    _BASE_BY_CODE.clear()
    _BASE_BY_NAME.clear()
    _BASE_BY_GLOTTOCODE.clear()
    _BASE_RETIRED.clear()


def _read_language_base(source: str, *, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Read the base from a path or URL, using the model cache for URLs (7 days)."""
    if not source.startswith(("http://", "https://")):
        path = Path(source[7:] if source.startswith("file://") else source)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: cannot read language base {path}: {exc}", file=sys.stderr)
            return None
    from .model_storage import read_model_cache_entry, write_model_cache_entry
    cached = read_model_cache_entry("language_base", max_age_seconds=7 * 86400)
    if isinstance(cached, dict) and "languages" in cached:
        return cached
    if not requests:
        return None
    try:
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] Warning: cannot load language base from {source}: {exc}", file=sys.stderr)
        return None
    if isinstance(data, dict) and "languages" in data:
        try:
            write_model_cache_entry("language_base", data)
        except (OSError, PermissionError):
            pass
        return data
    return None


def _load_language_base(*, verbose: bool = False) -> None:
    """Build the base indexes once per process (an empty base if it cannot be read)."""
    global _BASE_LOADED
    if _BASE_LOADED:
        return
    _BASE_LOADED = True  # also on failure: do not retry on every lookup
    import os
    data = _read_language_base(os.environ.get(LANGUAGE_BASE_ENV) or DEFAULT_LANGUAGE_BASE_URL, verbose=verbose)
    if not data:
        return
    names: Dict[str, List[Dict[str, Any]]] = {}
    for entry in data.get("languages") or []:
        if not isinstance(entry, dict):
            continue
        glotto = entry.get("glottocode")
        if glotto:
            _BASE_BY_GLOTTOCODE[glotto] = entry
            _BASE_BY_CODE.setdefault(glotto.lower(), entry)
        for key in ("iso_639_3", "iso_639_1", "iso_639_2t", "iso_639_2b"):
            code = entry.get(key)
            if code:
                _BASE_BY_CODE.setdefault(code.lower(), entry)
        # Names only for languages (and ISO-only codes): dialect and family names
        # are too often shared with languages to be safe as lookup keys.
        if entry.get("level") in ("language", "iso-only") or entry.get("iso_639_3"):
            for name in {entry.get("name"), entry.get("iso_name")}:
                if name:
                    names.setdefault(name.lower(), []).append(entry)
    for name, found in names.items():
        if len({id(e) for e in found}) == 1:  # ambiguous names are left out
            _BASE_BY_NAME[name] = found[0]
    retired = data.get("retired_iso_639_3")
    if isinstance(retired, dict):
        _BASE_RETIRED.update(retired)


def _base_entry(language: str) -> Optional[Dict[str, Any]]:
    """Exact code (ISO, Glottocode, retired ISO with one successor) or unique name in the base."""
    _load_language_base()
    key = language.lower().strip()
    # "Ho" is the Ho language, "ho" the ISO 639-1 code of Hiri Motu: input with
    # capitals is read as a name first.
    if language.strip() != key and key in _BASE_BY_NAME:
        return _BASE_BY_NAME[key]
    if key in _BASE_BY_CODE:
        return _BASE_BY_CODE[key]
    ret = _BASE_RETIRED.get(key)
    if ret and ret.get("change_to") and ret["change_to"].lower() in _BASE_BY_CODE:
        return _BASE_BY_CODE[ret["change_to"].lower()]
    return _BASE_BY_NAME.get(key)


def _base_iso_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The entry that carries the ISO codes: a dialect without ISO falls back to its language."""
    if not entry.get("iso_639_3") and entry.get("language") in _BASE_BY_GLOTTOCODE:
        return _BASE_BY_GLOTTOCODE[entry["language"]]
    return entry


def _base_lang_data(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The base entry in the shape of the hand table's lookup records."""
    iso = _base_iso_entry(entry)
    iso3 = iso.get("iso_639_3")
    if iso3 and iso3.lower() in _LANGUAGE_BY_CODE:  # the hand entry wins
        return _LANGUAGE_BY_CODE[iso3.lower()]
    return {
        "iso_639_1": iso.get("iso_639_1"),
        "iso_639_2": iso.get("iso_639_2t") or iso.get("iso_639_2b"),
        "iso_639_3": iso3,
        # ISO's reference name is the familiar one (Swiss German, not Central Alemannic)
        "primary_name": iso.get("iso_name") or iso.get("name"),
    }


def get_language_info(language: str) -> Dict[str, Any]:
    """
    Everything known about a language: the hand table's codes and name, plus the
    base's Glottocode, level, family, coordinates, macroarea, countries, ISO scope
    and type, and macrolanguage membership. Accepts ISO codes, Glottocodes and
    names. Returns {} when nothing matches.
    """
    _build_language_mappings()
    key = language.lower().strip()
    hand = _LANGUAGE_BY_CODE.get(key) or _LANGUAGE_BY_NAME.get(key)
    base = _base_entry(language)
    if hand and not base and hand.get("iso_639_3"):
        base = _BASE_BY_CODE.get(hand["iso_639_3"].lower())
    if not hand and not base:
        return {}
    info: Dict[str, Any] = {}
    if base:
        info.update({k: v for k, v in base.items() if k != "name"})
        info["glottolog_name"] = base.get("name")
        info.update({k: v for k, v in _base_lang_data(base).items() if v})
    if hand:
        info.update({k: v for k, v in hand.items() if v})
    return info


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
    
    # Exact code (ISO, Glottocode) or unique name in the generated Glottolog/ISO base
    base = _base_entry(language)
    if base is not None:
        lang_data = _base_lang_data(base)
        if lang_data.get("iso_639_3"):
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
    
    # Exact code (ISO, Glottocode) or unique name in the generated Glottolog/ISO base
    base = _base_entry(language)
    if base is not None:
        lang_data = _base_lang_data(base)
        if lang_data.get("iso_639_3"):
            return dict(lang_data)

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

