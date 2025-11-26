"""UDPipe REST backend implementation and registry spec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..backend_spec import BackendSpec
from ..conllu import conllu_to_document
from ..doc import Document
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
    clean_language_name,
    standardize_language_metadata,
)
from ..model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
)
from ..neural_backend import BackendManager, NeuralResult

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours
DEFAULT_REST_ENDPOINT = "https://lindat.mff.cuni.cz/services/udpipe/api/process"


def get_udpipe_model_entries(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Dict[str, str]]:
    cache_key = f"udpipe:{endpoint}"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            return cached

    prepared_models: Dict[str, Dict[str, str]] = {}
    try:
        response = requests.get(
            endpoint.replace("/process", "/models"),
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] Warning: failed to fetch UDPipe models: {exc}")
        payload = {}

    models_obj = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models_obj, dict):
        iterable = models_obj.items()
    elif isinstance(models_obj, list):
        iterable = ((item.get("name"), item) for item in models_obj if isinstance(item, dict))
    else:
        iterable = []

    for model_name, model_info in iterable:
        if not model_name:
            continue
        try:
            components: List[str]
            language_hint: Optional[str] = None
            if isinstance(model_info, dict):
                components = model_info.get("components") or model_info.get("available_components") or []
                language_hint = model_info.get("language") or model_info.get("lang") or model_info.get("abbr")
            elif isinstance(model_info, list):
                components = [str(comp) for comp in model_info]
            else:
                components = []

            slug = model_name.split("-ud-")[0]
            slug_parts = slug.split("-")
            primary_lang = slug_parts[0] if slug_parts else slug
            lang_candidate = (
                language_hint
                or primary_lang
                or model_name.split("_")[0]
            )
            lang_candidate = lang_candidate.replace("_", " ").replace("-", " ").strip()

            lang_metadata = standardize_language_metadata(language_code=None, language_name=lang_candidate)
            lang_code = lang_metadata.get(LANGUAGE_FIELD_ISO) or primary_lang[:2].lower()
            lang_display_final = clean_language_name(lang_metadata.get(LANGUAGE_FIELD_NAME) or lang_candidate.title())

            feature_parts: List[str] = []
            lowered_components = [comp.lower() for comp in components]
            if "tokenizer" in lowered_components:
                feature_parts.append("tokenization")
            if "tagger" in lowered_components:
                feature_parts.extend(["lemma", "upos", "xpos", "feats"])
            if "parser" in lowered_components:
                feature_parts.append("depparse")
            features = ", ".join(feature_parts) if feature_parts else "tokenization, tagging, parsing"

            entry = build_model_entry(
                "udpipe",
                model_name,
                language_code=lang_code,
                language_name=lang_display_final,
                features=features,
                components=components,
                preferred=lang_code == "en" and ("ewt" in model_name.lower() or "english" in model_name.lower()),
            )
            prepared_models[model_name] = entry
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: failed to parse UDPipe model '{model_name}': {exc}")
            continue

    if prepared_models:
        try:
            write_model_cache_entry(cache_key, prepared_models)
        except (OSError, PermissionError):
            pass
    return prepared_models


def list_udpipe_models_display(
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
    **kwargs: Any,
) -> int:
    try:
        prepared_models = get_udpipe_model_entries(
            url,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            verbose=verbose,
        )

        print(f"\nAvailable UDPipe models (deduplicated, showing newest versions only):")
        print(f"{'Model Name':<52} {'ISO':<8} {'Language':<20} {'Features':<30}")
        print("=" * 117)

        for model_name in sorted(prepared_models.keys()):
            entry = prepared_models[model_name]
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or ""
            lang_display = entry.get(LANGUAGE_FIELD_NAME) or ""
            features = entry.get("features", "unknown")
            suffix = "*" if entry.get("preferred") else ""
            display_name = f"{model_name}{suffix}"
            print(f"{display_name:<52} {lang_iso:<8} {lang_display:<20} {features:<30}")

        total_models = len(prepared_models)
        unique_languages = {
            entry.get(LANGUAGE_FIELD_ISO) or entry.get(LANGUAGE_FIELD_NAME)
            for entry in prepared_models.values()
            if entry.get(LANGUAGE_FIELD_ISO) or entry.get(LANGUAGE_FIELD_NAME)
        }
        print(f"\n(*) Preferred model used by auto-selection")
        print(f"Total: {total_models} model(s) for {len(unique_languages)} language(s)")

        return 0
    except Exception as e:
        print(f"Error fetching UDPipe models: {e}")
        import traceback

        traceback.print_exc()
        return 1


class UDPipeRESTBackend(BackendManager):
    """Neural backend that delegates tagging to a UDPipe REST service."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        model: Optional[str] = None,
        timeout: float = 30.0,
        batch_size: int = 50,
        extra_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Any = None,
        log_requests: bool = False,
    ):
        self._endpoint = endpoint_url
        self._model = model
        self._timeout = timeout
        self._batch_size = batch_size
        self._extra_params = extra_params or {}
        self._headers = headers or {}
        self._session = session or requests.Session()
        self._log = log_requests

    def _request(self, text: str) -> dict:
        payload = {"tokenizer": "", "tagger": "", "parser": "", "data": text}
        if "tokenizer" not in self._extra_params:
            payload["tokenizer"] = "normalized_spaces"
        payload.update(self._extra_params)
        if self._model:
            payload["model"] = self._model
        response = self._session.post(
            self._endpoint,
            data=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        del overrides, preserve_pos_tags, components
        text = "\n".join(sentence.text or "" for sentence in document.sentences)
        response = self._request(text)
        result_text = response.get("result", "")
        
        if not result_text:
            raise RuntimeError("UDPipe REST response did not contain any result data")
        
        # Parse the CoNLL-U result and return the tagged document
        tagged_doc = conllu_to_document(result_text, doc_id=document.id)
        
        # Preserve original metadata
        tagged_doc.meta.update(document.meta)
        
        return NeuralResult(document=tagged_doc, stats={})

    def train(self, *args, **kwargs):  # pragma: no cover - not implemented
        raise NotImplementedError("UDPipe REST backend does not support training.")

    def supports_training(self) -> bool:
        return False


def _create_udpipe_backend(
    *,
    model: str | None = None,
    model_name: str | None = None,
    language: str | None = None,
    endpoint_url: str | None = None,
    timeout: float = 30.0,
    batch_size: int = 50,
    extra_params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    session: Any = None,
    log_requests: bool = False,
    training: bool = False,
    **kwargs: Any,
) -> UDPipeRESTBackend:
    from ..backend_utils import validate_backend_kwargs, resolve_model_from_language
    
    validate_backend_kwargs(kwargs, "UDPipe", allowed_extra=["download_model", "training"])

    resolved_model = model or model_name
    if not resolved_model and language:
        resolved_model = resolve_model_from_language(language, "udpipe")
    
    if not resolved_model:
        raise ValueError("UDPipe REST backend requires a model name. Provide --model or --language.")

    resolved_endpoint = endpoint_url or DEFAULT_REST_ENDPOINT

    return UDPipeRESTBackend(
        resolved_endpoint,
        model=resolved_model,
        timeout=timeout,
        batch_size=batch_size,
        extra_params=extra_params,
        headers=headers,
        session=session,
        log_requests=log_requests,
    )


BACKEND_SPEC = BackendSpec(
    name="udpipe",
    description="UDPipe - REST service for UDPipe models (tokenization, tagging, parsing)",
    factory=_create_udpipe_backend,
    get_model_entries=get_udpipe_model_entries,
    list_models=list_udpipe_models_display,
    supports_training=False,
    is_rest=True,
)

