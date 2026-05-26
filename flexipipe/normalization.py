"""
Orthographic normalization utilities for flexipipe.

Delegates to FlexiNorm for profile-based normalization with confidence gating.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .doc import Document
from .segmentation import parse_segmenter_spec


def should_use_normalization(
    backend_name: str,
    document: Document,
    *,
    normalizer_spec: Optional[str] = None,
    norm_profile: Optional[str] = None,
    with_ocr: bool = False,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    auto_detect: bool = True,
) -> bool:
    """
    Determine if orthographic normalization should be used.
    """
    if normalizer_spec is not None:
        if normalizer_spec == "":
            return False
        return True

    if norm_profile:
        return True

    if with_ocr:
        return True

    if not auto_detect:
        return False

    if model_name and backend_name:
        try:
            from .model_catalog import build_unified_catalog
            catalog = build_unified_catalog(use_cache=True, verbose=False)
            model_key = f"{backend_name}:{model_name}"
            model_entry = catalog.get(model_key)
            if model_entry:
                requires_normalization = model_entry.get("requires_normalization")
                if requires_normalization is not None:
                    return bool(requires_normalization)
        except Exception:
            pass

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
    """Resolve normalizer specification to (backend, model) tuple."""
    if normalizer_spec and normalizer_spec != "":
        backend, model = parse_segmenter_spec(normalizer_spec)
        if verbose:
            model_str = f":{model}" if model else ""
            print(
                f"[flexipipe] Using explicit normalizer: {backend}{model_str}",
                file=__import__("sys").stderr,
            )
        return (backend, model)

    if normalizer_spec == "":
        if verbose:
            print("[flexipipe] Normalization explicitly disabled", file=__import__("sys").stderr)
        return None

    from .model_storage import get_default_normalizer

    default_normalizer = get_default_normalizer()
    if default_normalizer:
        backend, model = parse_segmenter_spec(default_normalizer)
        if verbose:
            model_str = f":{model}" if model else ""
            print(
                f"[flexipipe] Using configured default normalizer: {backend}{model_str}",
                file=__import__("sys").stderr,
            )
        return (backend, model)

    if model_name and backend_name:
        try:
            from .model_catalog import build_unified_catalog

            catalog = build_unified_catalog(use_cache=use_cache, verbose=False)
            model_key = f"{backend_name}:{model_name}"
            model_entry = catalog.get(model_key)
            if model_entry:
                preferred_normalizer = model_entry.get("preferred_normalizer")
                if preferred_normalizer:
                    backend, model = parse_segmenter_spec(preferred_normalizer)
                    if verbose:
                        model_str = f":{model}" if model else ""
                        print(
                            f"[flexipipe] Using registry-specified normalizer for model '{model_name}': {backend}{model_str}",
                            file=__import__("sys").stderr,
                        )
                    return (backend, model)
        except Exception:
            pass

    normalizer = find_normalizer_for_language(language, use_cache=use_cache)
    if normalizer:
        if verbose:
            backend, model = normalizer
            model_str = f":{model}" if model else ""
            print(
                f"[flexipipe] Auto-detected normalizer for '{language}': {backend}{model_str}",
                file=__import__("sys").stderr,
            )
        return normalizer

    return ("hunspell", None)


def find_normalizer_for_language(
    language: Optional[str],
    *,
    use_cache: bool = True,
) -> Optional[Tuple[str, Optional[str]]]:
    """Find an available normalizer for a language."""
    if not language:
        return None
    try:
        import phunspell

        lang_code = _normalize_language_for_phunspell(language)
        if lang_code:
            return ("hunspell", lang_code)
    except ImportError:
        pass
    except Exception:
        pass
    return None


def _normalize_language_for_phunspell(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    lang_lower = language.lower().replace("-", "_")
    if "_" in lang_lower and len(lang_lower.split("_")) == 2:
        return lang_lower
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
    if lang_lower in lang_mapping:
        return lang_mapping[lang_lower]
    if len(lang_lower) == 2:
        return f"{lang_lower}_{lang_lower.upper()}"
    return lang_lower


def apply_normalization(
    document: Document,
    normalizer: Tuple[str, Optional[str]],
    *,
    language: Optional[str] = None,
    profile: str = "typo",
    mode: str = "light",
    subtype: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    rules_path: Optional[str] = None,
    flexitag_params: Optional[str] = None,
    multilayered_model_path: Optional[str] = None,
    with_ocr: bool = False,
    ocr_lexicon_path: Optional[str] = None,
    ocr_rules_path: Optional[str] = None,
    ocr_model_name: Optional[str] = None,
    jamspell_model_path: Optional[str] = None,
    verbose: bool = False,
) -> Document:
    """
    Apply normalization via FlexiNorm.

    Sets token.reg (and other layers) without modifying token.form.
    """
    if isinstance(normalizer, str):
        backend_name, model_name = parse_segmenter_spec(normalizer)
    else:
        backend_name, model_name = normalizer

    if verbose:
        model_str = f" with model '{model_name}'" if model_name else ""
        print(
            f"[flexipipe] FlexiNorm profile={profile} mode={mode} backend={backend_name}{model_str}",
            file=__import__("sys").stderr,
        )

    try:
        from flexinorm import NormalizationPipeline
        from flexinorm.profiles import HistoricSubtype, NormalizationProfile
    except ImportError as e:
        raise ImportError(
            "FlexiNorm is required for normalization. Install with: pip install -e /path/to/flexinorm"
        ) from e

    norm_profile = profile
    if profile == "typo" and backend_name not in ("typo",):
        pass  # keep typo profile, backend from normalizer

    subtype_enum = None
    if subtype:
        subtype_enum = HistoricSubtype(subtype)

    profile_arg = norm_profile if isinstance(norm_profile, str) else NormalizationProfile(norm_profile)
    pipeline = NormalizationPipeline.from_profile(
        profile_arg,
        mode=mode,
        subtype=subtype_enum,
        language=language or model_name or "en",
        normalizer_backend=backend_name,
        normalizer_model=model_name,
        lexicon_path=lexicon_path,
        rules_path=rules_path,
        flexitag_params=flexitag_params,
        multilayered_model_path=multilayered_model_path,
        with_ocr=with_ocr,
        ocr_lexicon_path=ocr_lexicon_path,
        ocr_rules_path=ocr_rules_path,
        ocr_model_name=ocr_model_name,
        jamspell_model_path=jamspell_model_path,
    )
    pipeline.normalize(document)

    tokens_normalized = any(
        t.reg or t.get_attr("reg_confidence")
        for s in document.sentences
        for t in s.tokens
    )
    if tokens_normalized:
        document.meta["_normalization_applied"] = True
        document.meta["_normalization_engine"] = "flexinorm"
        document.meta["norm_profile"] = profile
        document.meta["norm_mode"] = mode

    return document


def maybe_apply_normalization(
    document: Document,
    backend_name: str,
    *,
    normalizer_spec: Optional[str] = None,
    norm_profile: Optional[str] = None,
    with_ocr: bool = False,
    norm_mode: str = "light",
    norm_subtype: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    rules_path: Optional[str] = None,
    flexitag_params: Optional[str] = None,
    ocr_lexicon_path: Optional[str] = None,
    ocr_rules_path: Optional[str] = None,
    ocr_model_name: Optional[str] = None,
    jamspell_model_path: Optional[str] = None,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> Tuple[Document, bool]:
    """Apply normalization if needed."""
    profile = norm_profile or "typo"

    if not should_use_normalization(
        backend_name,
        document,
        normalizer_spec=normalizer_spec,
        norm_profile=norm_profile,
        with_ocr=with_ocr,
        language=language,
        model_name=model_name,
        auto_detect=norm_profile is None and not with_ocr,
    ):
        return document, False

    resolved_normalizer = resolve_normalizer_spec(
        normalizer_spec,
        language,
        backend_name=backend_name,
        model_name=model_name,
        use_cache=use_cache,
        verbose=verbose,
    )

    if not resolved_normalizer and not norm_profile:
        return document, False

    if not resolved_normalizer:
        resolved_normalizer = ("hunspell", language)

    normalized_doc = apply_normalization(
        document,
        resolved_normalizer,
        language=language,
        profile=profile,
        mode=norm_mode,
        subtype=norm_subtype,
        lexicon_path=lexicon_path,
        rules_path=rules_path,
        flexitag_params=flexitag_params,
        with_ocr=with_ocr,
        ocr_lexicon_path=ocr_lexicon_path,
        ocr_rules_path=ocr_rules_path,
        ocr_model_name=ocr_model_name,
        jamspell_model_path=jamspell_model_path,
        verbose=verbose,
    )

    return normalized_doc, True
