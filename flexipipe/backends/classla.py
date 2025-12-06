"""Backend and registry spec for ClassLA."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend_spec import BackendSpec
from ..doc import Document, Entity, Sentence, Token
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
)
from ..model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
)
from ..neural_backend import BackendManager, NeuralResult

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def get_classla_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    **kwargs: Any,
) -> Dict[str, Dict[str, str]]:
    del kwargs
    cache_key = "classla"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached:
            fixed = False
            for model_key, entry in cached.items():
                if isinstance(entry, dict) and not entry.get("language_iso") and "-" in model_key:
                    entry["language_iso"] = model_key.split("-")[0].lower()
                    fixed = True
            if fixed and refresh_cache:
                try:
                    write_model_cache_entry(cache_key, cached)
                except (OSError, PermissionError):
                    pass
            if cache_entries_standardized(cached):
                return cached

    result: Dict[str, Dict[str, str]] = {}
    installed_models: Dict[str, Dict[str, str]] = {}
    classla_resources = get_backend_models_dir("classla", create=False)
    if classla_resources.exists():
        for lang_dir in classla_resources.iterdir():
            if not lang_dir.is_dir():
                continue
            lang_code = lang_dir.name
            for processor_dir in lang_dir.iterdir():
                if not processor_dir.is_dir():
                    continue
                for model_file in processor_dir.glob("*.pt"):
                    package = model_file.stem
                    variant = "nonstandard" if "nonstandard" in processor_dir.parts else "standard"
                    model_key = f"{lang_code}-{variant}"
                    installed_models.setdefault(
                        model_key,
                        {"lang": lang_code, "package": package, "variant": variant},
                    )

    known_models = {
        ("hr", "standard"): {"package": "set", "name": "Croatian"},
        ("hr", "nonstandard"): {"package": "set", "name": "Croatian (nonstandard)"},
        ("sr", "standard"): {"package": "set", "name": "Serbian"},
        ("sr", "nonstandard"): {"package": "set", "name": "Serbian (nonstandard)"},
        ("bg", "standard"): {"package": "btb", "name": "Bulgarian"},
        ("mk", "standard"): {"package": "mk", "name": "Macedonian"},
        ("sl", "standard"): {"package": "ssj", "name": "Slovenian"},
    }

    for (lang_code, variant), model_info in known_models.items():
        package = model_info["package"]
        model_key = f"{lang_code}-{variant}"
        model_entry = installed_models.get(
            model_key,
            {"lang": lang_code, "package": package, "variant": variant},
        )
        lang_name = model_info.get("name", lang_code.upper())
        entry = build_model_entry(
            backend="classla",
            model_id=model_key,
            model_name=model_key,
            language_code=model_entry.get("lang", lang_code),
            language_name=lang_name,
            package=model_entry.get("package", package),
            description=f"ClassLA model for {lang_name} ({variant})",
        )
        entry["language_iso"] = entry.get("language_iso") or lang_code.lower()
        entry["package"] = model_entry.get("package", package)
        entry["variant"] = model_entry.get("variant", variant)
        result[model_key] = entry

    if refresh_cache:
        try:
            write_model_cache_entry(cache_key, result)
        except (OSError, PermissionError):
            pass
    return result


def _classla_doc_to_document(
    classla_doc,
    original_doc: Optional[Document] = None,
) -> Document:
    doc = Document(id="")
    if original_doc:
        doc.id = original_doc.id
        doc.meta = original_doc.meta.copy()
        doc.attrs = original_doc.attrs.copy()

    for stanza_sent in classla_doc.sentences:
        sent_id = getattr(stanza_sent, "sent_id", None) or ""
        sentence = Sentence(id=sent_id, sent_id=sent_id, text=stanza_sent.text, tokens=[])

        if hasattr(stanza_sent, "ents") and stanza_sent.ents:
            for ent in stanza_sent.ents:
                try:
                    token_start = getattr(ent, "start", None)
                    token_end = getattr(ent, "end", None)
                    if hasattr(ent, "tokens") and ent.tokens:
                        token_start = ent.tokens[0].id
                        token_end = ent.tokens[-1].id
                    if token_start is None or token_end is None:
                        continue
                    start_idx = int(str(token_start).split("-")[0])
                    end_idx = int(str(token_end).split("-")[-1])
                    entity = Entity(
                        start=start_idx + 1,
                        end=end_idx,
                        label=getattr(ent, "type", ""),
                        text=getattr(ent, "text", ""),
                    )
                    sentence.entities.append(entity)
                except Exception:
                    continue

        for token_idx, stanza_token in enumerate(stanza_sent.tokens):
            def _to_int_id(token_id):
                if token_id is None:
                    return token_idx + 1
                token_str = str(token_id)
                if "-" in token_str:
                    token_str = token_str.split("-")[0]
                try:
                    return int(token_str)
                except (ValueError, TypeError):
                    return token_idx + 1

            if hasattr(stanza_token, "words") and len(stanza_token.words) > 1:
                subtokens = []
                for word in stanza_token.words:
                    subtokens.append(
                        Token(
                            id=_to_int_id(word.id),
                            form=word.text,
                            lemma=word.lemma or "",
                            upos=word.upos or "",
                            xpos=word.xpos or "",
                            feats=word.feats or "",
                            head=word.head if word.head else 0,
                            deprel=word.deprel or "",
                            space_after=("SpaceAfter=No" not in (word.misc or "")) if word.misc else True,
                        )
                    )
                token = Token(
                    id=_to_int_id(stanza_token.id),
                    form=stanza_token.text,
                    lemma=stanza_token.words[0].lemma if stanza_token.words else "",
                    upos=stanza_token.words[0].upos if stanza_token.words else "",
                    xpos=stanza_token.words[0].xpos if stanza_token.words else "",
                    feats=stanza_token.words[0].feats if stanza_token.words else "",
                    head=stanza_token.words[0].head if stanza_token.words and stanza_token.words[0].head else 0,
                    deprel=stanza_token.words[0].deprel if stanza_token.words else "",
                    is_mwt=True,
                    subtokens=subtokens,
                    space_after=("SpaceAfter=No" not in (stanza_token.misc or "")) if stanza_token.misc else True,
                )
                token.parts = [st.form for st in subtokens]
            else:
                word = stanza_token.words[0] if hasattr(stanza_token, "words") and stanza_token.words else stanza_token
                token = Token(
                    id=_to_int_id(stanza_token.id),
                    form=stanza_token.text,
                    lemma=getattr(word, "lemma", "") or "",
                    upos=getattr(word, "upos", "") or "",
                    xpos=getattr(word, "xpos", "") or "",
                    feats=getattr(word, "feats", "") or "",
                    head=getattr(word, "head", 0) or 0,
                    deprel=getattr(word, "deprel", "") or "",
                    space_after=("SpaceAfter=No" not in (stanza_token.misc or "")) if getattr(stanza_token, "misc", None) else True,
                )
            sentence.tokens.append(token)

        if sentence.tokens:
            sentence.tokens[-1].space_after = None
        doc.sentences.append(sentence)
    return doc


class ClassLABackend(BackendManager):
    """ClassLA-based neural backend."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        package: Optional[str] = None,
        processors: Optional[str] = None,
        use_gpu: bool = False,
        download_model: bool = False,
        verbose: bool = False,
        type: Optional[str] = None,
    ):
        from ..model_storage import setup_backend_environment

        setup_backend_environment("classla")

        if not verbose:
            logging.getLogger("classla").setLevel(logging.WARNING)

        try:
            import classla
        except ImportError as exc:
            raise ImportError("ClassLA backend requires the 'classla' package. Install it with: pip install classla") from exc
        try:
            from classla.pipeline.core import ResourcesFileNotFound  # type: ignore
        except ImportError:
            ResourcesFileNotFound = Exception

        self.classla = classla
        self._resources_error = ResourcesFileNotFound
        self._language = language or (model_name.split("-")[0] if model_name and "-" in model_name else model_name) or "hr"
        default_package = {"hr": "set", "sr": "set", "bg": "btb", "mk": "mk", "sl": "ssj"}
        self._package = package or default_package.get(self._language)
        self._processors = processors or "tokenize,pos,lemma"
        self._use_gpu = use_gpu
        self._download = download_model
        self._verbose = verbose
        if model_name and "-" in model_name and not type:
            parts = model_name.split("-", 1)
            if len(parts) == 2 and parts[1] in ("standard", "nonstandard"):
                self._type = parts[1]
            else:
                self._type = type or "standard"
        else:
            self._type = type or "standard"
        self._pipelines: Dict[bool, classla.Pipeline] = {}

    def _build_pipeline(self, pretokenized: bool):
        if not self._verbose:
            classla_logger = logging.getLogger("classla")
            classla_logger.setLevel(logging.WARNING)
            for handler in classla_logger.handlers:
                handler.setLevel(logging.WARNING)
            classla_logger.propagate = False

        from ..model_storage import get_backend_models_dir

        classla_dir = get_backend_models_dir("classla", create=False)

        config: Dict[str, Union[str, bool]] = {
            "lang": self._language,
            "processors": self._processors,
            "use_gpu": self._use_gpu,
            "type": self._type,
            "dir": str(classla_dir),
        }
        if pretokenized:
            config["tokenize_pretokenized"] = True

        try:
            return self.classla.Pipeline(**config)
        except (ValueError, TypeError) as e:
            error_str = str(e)
            if "not enough values to unpack" in error_str or "expected 2" in error_str:
                config_minimal = {
                    "lang": self._language,
                    "processors": self._processors,
                    "use_gpu": self._use_gpu,
                    "type": self._type,
                }
                pipeline = self.classla.Pipeline(**config_minimal)
                if pretokenized and hasattr(pipeline, "tokenize_pretokenized"):
                    pipeline.tokenize_pretokenized = True
                return pipeline
            raise
        except (FileNotFoundError, OSError) as e:
            error_str = str(e).lower()
            if "depparse" in error_str or "parser" in error_str or "no such file" in error_str:
                processors_list = [p.strip() for p in self._processors.split(",") if p.strip()]
                if "depparse" in processors_list:
                    processors_list.remove("depparse")
                    config_fallback = dict(config)
                    config_fallback["processors"] = ",".join(processors_list)
                    return self.classla.Pipeline(**config_fallback)
            if self._download:
                import sys
                print(f"[flexipipe] Downloading ClassLA model for {self._language} (type: {self._type})...", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                # Pass verbose=True to classla.download to ensure progress is shown
                # In non-interactive environments, this helps with output flushing
                self.classla.download(self._language, type=self._type, verbose=True)
                sys.stdout.flush()
                sys.stderr.flush()
                print(f"[flexipipe] ClassLA model download completed", flush=True)
                return self.classla.Pipeline(**config)
            raise
        except self._resources_error as e:
            if self._download:
                import sys
                print(f"[flexipipe] Downloading ClassLA model for {self._language} (type: {self._type})...", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                # Pass verbose=True to classla.download to ensure progress is shown
                # In non-interactive environments, this helps with output flushing
                self.classla.download(self._language, type=self._type, verbose=True)
                sys.stdout.flush()
                sys.stderr.flush()
                print(f"[flexipipe] ClassLA model download completed", flush=True)
                return self.classla.Pipeline(**config)
            raise RuntimeError(
                f"ClassLA model not found for language '{self._language}' "
                f"(package: {self._package}, type: {self._type}). "
                f"Install it with: classla.download('{self._language}', type='{self._type}')"
            ) from e

    def _get_pipeline(self, pretokenized: bool):
        if pretokenized not in self._pipelines:
            self._pipelines[pretokenized] = self._build_pipeline(pretokenized)
        return self._pipelines[pretokenized]

    def _run_raw(self, document: Document):
        pipeline = self._get_pipeline(pretokenized=False)
        text = "\n".join(sent.text for sent in document.sentences if sent.text)
        return pipeline(text)

    def _run_pretokenized(self, document: Document):
        pipeline = self._get_pipeline(pretokenized=True)
        pretokenized = [[token.form for token in sentence.tokens] for sentence in document.sentences]
        return pipeline(pretokenized)

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        del overrides, preserve_pos_tags, components

        start_time = time.time()
        if use_raw_text or not document.sentences:
            classla_doc = self._run_raw(document)
            result_doc = _classla_doc_to_document(classla_doc)
        else:
            classla_doc = self._run_pretokenized(document)
            result_doc = _classla_doc_to_document(classla_doc, original_doc=document)

        elapsed = time.time() - start_time
        token_count = sum(len(sent.tokens) for sent in result_doc.sentences)
        stats = {
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
            "sentences_per_second": len(result_doc.sentences) / elapsed if elapsed > 0 else 0.0,
        }
        return NeuralResult(document=result_doc, stats=stats)

    def train(
        self,
        train_data: Union[Document, List[Document], Path],
        output_dir: Path,
        *,
        dev_data: Optional[Union[Document, List[Document], Path]] = None,
        **kwargs,
    ) -> Path:
        raise NotImplementedError(
            "ClassLA training is not yet integrated into flexipipe. "
            "Use the official classla-train workflow to build models."
        )

    def supports_training(self) -> bool:
        return False


def _list_classla_models(*args, **kwargs) -> int:
    entries = get_classla_model_entries(*args, **kwargs)
    print(f"\nAvailable ClassLA models:")
    print(f"{'Model ID':<20} {'ISO':<6} {'Language':<20} {'Package':<15} {'Variant':<15} {'Status':<25}")
    print("=" * 101)
    for key in sorted(entries.keys()):
        model_info = entries[key]
        model_id = key
        iso = (model_info.get("language_iso") or "")[:6]
        lang = model_info.get("language_name", "")
        pkg = model_info.get("package", "")
        variant = model_info.get("variant", "standard")
        status = model_info.get("status", "Available")
        print(f"{model_id:<20} {iso:<6} {lang:<20} {pkg:<15} {variant:<15} {status:<25}")
    print(f"\nTotal: {len(entries)} model(s)")
    print("\nClassLA models are downloaded automatically on first use")
    print("Features: tokenization, lemma, upos, xpos, feats, depparse, NER")
    print("\nUsage: --backend classla --model <Model ID>")
    return 0


def _create_classla_backend(
    *,
    model_name: str | None = None,
    language: str | None = None,
    package: str | None = None,
    processors: str | None = None,
    use_gpu: bool = False,
    download_model: bool = False,
    verbose: bool = False,
    type: str | None = None,
    training: bool = False,
    **kwargs: Any,
) -> ClassLABackend:
    _ = training
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unexpected ClassLA backend arguments: {unexpected}")
    return ClassLABackend(
        model_name=model_name,
        language=language,
        package=package,
        processors=processors,
        use_gpu=use_gpu,
        download_model=download_model,
        verbose=verbose,
        type=type,
    )


BACKEND_SPEC = BackendSpec(
    name="classla",
    description="ClassLA - Fork of Stanza for South Slavic languages",
    factory=_create_classla_backend,
    get_model_entries=get_classla_model_entries,
    list_models=_list_classla_models,
    supports_training=False,
    is_rest=False,
    url="https://github.com/clarinsi/classla",
)

