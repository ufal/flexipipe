from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, Optional

from ..language_detector_registry import LanguageDetectorSpec, register_language_detector
from ..model_storage import get_flexipipe_models_dir


# Lightweight trigram-based detector (CWALI-style).
# Models are expected at: <models_dir>/language-detector/trigram_models.json
# File format:
# {
#   "en": { "the": 1234, "and": 987, ... },
#   "nl": { ... }
# }
# Counts can be raw; we normalize at runtime.

_TRIGRAM_MODELS: Dict[str, Dict[str, float]] | None = None
_TRIGRAM_META: Dict[str, Dict[str, str]] | None = None
_TRIGRAM_MODELS_URL = (
    "https://raw.githubusercontent.com/ufal/flexipipe-models/refs/heads/main"
    "/resources/trigram_models.json"
)


def _maybe_download_trigram_models(models_path: Path, verbose: bool) -> None:
    """Download trigram models from the upstream flexipipe-models repo if absent."""
    if models_path.exists():
        return
    models_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[flexipipe] trigram detector: downloading models from {_TRIGRAM_MODELS_URL}")
    try:
        with urllib.request.urlopen(_TRIGRAM_MODELS_URL, timeout=15) as resp:
            content = resp.read()
        models_path.write_bytes(content)
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] trigram detector: download failed: {exc}")


def _load_trigram_models(verbose: bool = False) -> Dict[str, Dict[str, float]]:
    global _TRIGRAM_MODELS
    global _TRIGRAM_META
    if _TRIGRAM_MODELS is not None:
        return _TRIGRAM_MODELS
    models_path = (
        get_flexipipe_models_dir(create=False)
        / "language-detector"
        / "trigram_models.json"
    )
    if not models_path.exists():
        _maybe_download_trigram_models(models_path, verbose)
    if not models_path.exists():
        if verbose:
            print(
                "[flexipipe] trigram detector: models file not found at "
                f"{models_path}. Place trigram_models.json there to enable."
            )
        _TRIGRAM_MODELS = {}
        _TRIGRAM_META = {}
        return _TRIGRAM_MODELS
    try:
        with models_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            models: Dict[str, Dict[str, float]] = {}
            meta: Dict[str, Dict[str, str]] = {}

            def _process_entry(code_key: str, payload: Any) -> None:
                # Two supported shapes:
                # 1) {"iso": "...", "name": "...", "code": "...", "trigrams": {...}}
                # 2) {"tri": weight, ...} with code_key as the language code
                if isinstance(payload, dict) and "trigrams" in payload:
                    trigrams = payload.get("trigrams", {})
                    iso = str(payload.get("iso") or code_key).strip()
                    code = str(payload.get("code") or iso).strip().lower()
                    name = str(payload.get("name") or code).strip()
                    # In CWALI data, the values in `trigrams` are already ranks
                    if not isinstance(trigrams, dict):
                        return
                    rank_map: Dict[str, float] = {tri: float(v) for tri, v in trigrams.items()}
                else:
                    # Legacy/simple shape: { trigram: weight }, convert weights to ranks
                    trigrams = payload
                    iso = str(code_key).strip()
                    code = iso.lower()
                    name = iso
                    if not isinstance(trigrams, dict):
                        return
                    sorted_items = sorted(trigrams.items(), key=lambda kv: kv[1])
                    rank_map = {tri: float(idx) for idx, (tri, _) in enumerate(sorted_items)}

                models[code] = rank_map
                meta[code] = {"iso": iso.lower(), "code": code, "name": name}

            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    code_key = entry.get("code") or entry.get("iso") or entry.get("name")
                    if not code_key:
                        continue
                    _process_entry(str(code_key), entry)
            elif isinstance(data, dict):
                for code_key, payload in data.items():
                    _process_entry(str(code_key), payload)
            else:
                raise ValueError("Invalid format: expected object or list at top level")

            _TRIGRAM_MODELS = models
            _TRIGRAM_META = meta
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] trigram detector: failed to load models: {exc}")
        _TRIGRAM_MODELS = {}
        _TRIGRAM_META = {}
    return _TRIGRAM_MODELS


def _create_ordered_model(text: str) -> list[tuple[str, float]]:
    # Raw counts, sorted descending (to mirror the original JS logic)
    counts: Dict[str, int] = {}
    lowered = text.lower()
    for i in range(len(lowered) - 2):
        trigram = lowered[i : i + 3]
        counts[trigram] = counts.get(trigram, 0) + 1
    items = [(k, float(v)) for k, v in counts.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items


def _compare(model: list[tuple[str, float]], known: Dict[str, float]) -> float:
    # Mirror CWALI JS: distance on counts vs rank, unseen trigram penalty of 300.
    if not known or not model:
        return 0.0
    dist = 0.0
    penalty = 300.0
    for trigram, count in model:
        known_rank = known.get(trigram)
        if known_rank is None:
            dist += penalty
        else:
            dist += abs(count - known_rank)
    score = 1.0 - (dist / (penalty * len(model)))
    return max(0.0, score)


def _detect_trigram(
    text: str,
    min_length: int,
    confidence_threshold: float,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    if not text or len(text.strip()) < min_length:
        if verbose:
            print("[flexipipe] trigram detector skipped: input too short.")
        return None
    models = _load_trigram_models(verbose=verbose)
    meta = _TRIGRAM_META or {}
    if not models:
        return None
    model = _create_ordered_model(text)
    best_lang: Optional[str] = None
    best_score = 0.0
    candidates: list[tuple[str, float]] = []
    for lang, profile in models.items():
        score = _compare(model, profile)
        candidates.append((lang, score))
        if score > best_score:
            best_score = score
            best_lang = lang
    if best_lang is None or best_score <= 0.0:
        return None
    # Sort candidates by score, descending
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    # CWALI-style reliability: second-best must be <= 80% of best
    second_score = candidates[1][1] if len(candidates) > 1 else 0.0
    separation_ratio = (second_score / best_score) if best_score > 0 else 1.0
    is_reliable = separation_ratio <= 0.80

    lang_meta = meta.get(best_lang, {})
    language_iso = lang_meta.get("iso") or best_lang
    language_name = lang_meta.get("name") or language_iso
    result = {
        "language": best_lang,
        "language_iso": language_iso,
        "language_name": language_name,
        "confidence": best_score,
        "separation_ratio": separation_ratio,
        "reliable": is_reliable,
    }
    if verbose:
        result["candidates"] = [
            {
                "language_iso": meta.get(lang, {}).get("iso") or lang,
                "language_name": meta.get(lang, {}).get("name") or lang,
                "confidence": score,
            }
            for lang, score in candidates[:5]
        ]
    # Confidence threshold is handled by the caller for trigram; we always return
    return result


# Register detector
register_language_detector(
    LanguageDetectorSpec(
        name="trigram",
        description="Lightweight trigram-based detector (requires trigram_models.json)",
        detect=_detect_trigram,
        prepare=None,
        is_default=False,
    )
)


