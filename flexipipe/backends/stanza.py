"""Stanza backend implementation and registry spec."""

from __future__ import annotations

import logging
import os
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
    clean_language_name,
)
from ..model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
    setup_backend_environment,
)
from ..neural_backend import BackendManager, NeuralResult

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours
DEFAULT_STANZA_PACKAGES = {
    "en": "ewt",
    "cs": "pdt",
    "de": "gsd",
    "fr": "gsd",
    "es": "gsd",
    "it": "isdt",
    "ru": "syntagrus",
    "zh": "gsd",
    "ja": "gsd",
    "ko": "gsd",
    "ar": "padt",
    "hi": "hdtdtb",
    "vi": "vtb",
    "tr": "imst",
    "pl": "pdb",
    "uk": "iu",
    "bg": "btb",
    "hr": "set",
    "sr": "set",
    "sk": "snk",
    "sl": "ssj",
    "pt": "bosque",
    "ro": "rrt",
    "fi": "tdt",
    "sv": "talbanken",
    "no": "bokmaal",
    "da": "ddt",
    "nl": "alpino",
    "et": "edt",
    "lv": "lvtb",
    "lt": "alksnys",
    "el": "gdt",
    "he": "htb",
    "fa": "seraji",
    "id": "gsd",
    "th": "pud",
    "ta": "ttb",
    "te": "mtg",
    "ur": "udtb",
    "eu": "bdt",
    "ga": "idt",
    "cy": "ccg",
    "gd": "arcosg",
    "br": "keb",
    "mt": "mudt",
    "is": "icepahc",
    "fo": "farpahc",
    "nn": "nynorsk",
    "be": "hse",
    "kk": "ktb",
    "hy": "armtdp",
    "mr": "ufal",
    "sa": "vedic",
    "la": "ittb",
    "grc": "proiel",
    "cu": "proiel",
    "got": "proiel",
    "ug": "udt",
    "bn": "iitb",
    "km": "wtb",
    "lo": "pud",
    "my": "myu",
}


def get_stanza_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
) -> Dict[str, Dict[str, str]]:
    cache_key = "stanza"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            return cached

    result: Dict[str, Dict[str, str]] = {}

    installed_models = {}
    try:
        stanza_resources = get_backend_models_dir("stanza", create=False)
    except (OSError, PermissionError):
        return result
    if stanza_resources.exists():
        try:
            for lang_dir in stanza_resources.iterdir():
                if lang_dir.is_dir() and not lang_dir.name.startswith("."):
                    lang = lang_dir.name
                    for processor_dir in lang_dir.iterdir():
                        if processor_dir.is_dir() and not processor_dir.name.startswith("."):
                            pt_files = list(processor_dir.glob("*.pt"))
                            for pt_file in pt_files:
                                pkg = pt_file.stem
                                key = f"{lang}_{pkg}"
                                if key not in installed_models:
                                    mtime = pt_file.stat().st_mtime
                                    date_str = time.strftime("%Y-%m-%d", time.localtime(mtime))
                                    installed_models[key] = date_str
        except (OSError, PermissionError):
            pass

        def _extract_model_from_pt_file(pt_file: Path, default_lang: str = None, default_pkg: str = None) -> Optional[tuple]:
            filename = pt_file.stem
            for suffix in [
                "_nocharlm_tokenizer",
                "_tokenizer",
                "_tagger",
                "_lemmatizer",
                "_parser",
                "_pos",
                "_lemma",
                "_depparse",
            ]:
                if filename.endswith(suffix):
                    filename = filename[: -len(suffix)]
                    break
            parts = filename.split("_", 2)
            if len(parts) >= 2:
                return parts[0], parts[1]
            if default_lang and default_pkg:
                return default_lang, default_pkg
            return None

        for subdir in stanza_resources.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                models_dir = subdir / "models"
                if models_dir.exists():
                    subdir_parts = subdir.name.split("_")
                    default_lang = subdir_parts[0] if len(subdir_parts) > 0 else None
                    default_pkg = subdir_parts[1] if len(subdir_parts) > 1 else None

                    for comp_dir in models_dir.iterdir():
                        if comp_dir.is_dir() and not comp_dir.name.startswith("."):
                            pt_files = list(comp_dir.glob("*.pt"))
                            for pt_file in pt_files:
                                result_entry = _extract_model_from_pt_file(pt_file, default_lang, default_pkg)
                                if result_entry:
                                    lang, pkg = result_entry
                                    key = f"{lang}_{pkg}"
                                    if key not in installed_models:
                                        mtime = pt_file.stat().st_mtime
                                        date_str = time.strftime("%Y-%m-%d", time.localtime(mtime))
                                        installed_models[key] = date_str

                    for shorthand_dir in models_dir.iterdir():
                        if (
                            shorthand_dir.is_dir()
                            and not shorthand_dir.name.startswith(".")
                            and "_" in shorthand_dir.name
                        ):
                            parts = shorthand_dir.name.split("_", 1)
                            if len(parts) == 2:
                                lang, pkg = parts
                                pt_files = list(shorthand_dir.glob("*.pt"))
                                if pt_files:
                                    key = f"{lang}_{pkg}"
                                    if key not in installed_models:
                                        mtime = max(f.stat().st_mtime for f in pt_files)
                                        date_str = time.strftime("%Y-%m-%d", time.localtime(mtime))
                                        installed_models[key] = date_str

        models_dir_root = stanza_resources / "models"
        if models_dir_root.exists():
            for comp_dir in models_dir_root.iterdir():
                if comp_dir.is_dir() and not comp_dir.name.startswith("."):
                    pt_files = list(comp_dir.glob("*.pt"))
                    for pt_file in pt_files:
                        result_entry = _extract_model_from_pt_file(pt_file)
                        if result_entry:
                            lang, pkg = result_entry
                            key = f"{lang}_{pkg}"
                            if key not in installed_models:
                                mtime = pt_file.stat().st_mtime
                                date_str = time.strftime("%Y-%m-%d", time.localtime(mtime))
                                installed_models[key] = date_str

    common_models = [
        ("en", "ewt", "English (EWT)"),
        ("en", "gum", "English (GUM)"),
        ("cs", "pdt", "Czech (PDT)"),
        ("cs", "cac", "Czech (CAC)"),
        ("de", "gsd", "German (GSD)"),
        ("fr", "gsd", "French (GSD)"),
        ("es", "gsd", "Spanish (GSD)"),
        ("it", "isdt", "Italian (ISDT)"),
        ("ru", "syntagrus", "Russian (SynTagRus)"),
        ("zh", "gsd", "Chinese (GSD)"),
        ("ja", "gsd", "Japanese (GSD)"),
        ("ko", "gsd", "Korean (GSD)"),
        ("ar", "padt", "Arabic (PADT)"),
        ("hi", "hdtdtb", "Hindi (HDTDTB)"),
        ("vi", "vtb", "Vietnamese (VTB)"),
        ("tr", "imst", "Turkish (IMST)"),
        ("pl", "pdb", "Polish (PDB)"),
        ("uk", "iu", "Ukrainian (IU)"),
        ("bg", "btb", "Bulgarian (BTB)"),
        ("hr", "set", "Croatian (SET)"),
        ("sr", "set", "Serbian (SET)"),
        ("sk", "snk", "Slovak (SNK)"),
        ("sl", "ssj", "Slovenian (SSJ)"),
        ("ca", "ancora", "Catalan (AnCora)"),
        ("gl", "treegal", "Galician (TreeGal)"),
        ("pt", "bosque", "Portuguese (Bosque)"),
        ("ro", "rrt", "Romanian (RRT)"),
        ("fi", "tdt", "Finnish (TDT)"),
        ("sv", "talbanken", "Swedish (Talbanken)"),
        ("no", "bokmaal", "Norwegian Bokmål"),
        ("da", "ddt", "Danish (DDT)"),
        ("nl", "alpino", "Dutch (Alpino)"),
        ("et", "edt", "Estonian (EDT)"),
        ("lv", "lvtb", "Latvian (LVTB)"),
        ("lt", "alksnys", "Lithuanian (ALKSNYS)"),
        ("el", "gdt", "Greek (GDT)"),
        ("he", "htb", "Hebrew (HTB)"),
        ("fa", "seraji", "Persian (Seraji)"),
        ("id", "gsd", "Indonesian (GSD)"),
        ("th", "pud", "Thai (PUD)"),
        ("ta", "ttb", "Tamil (TTB)"),
        ("te", "mtg", "Telugu (MTG)"),
        ("ur", "udtb", "Urdu (UDTB)"),
        ("eu", "bdt", "Basque (BDT)"),
        ("ga", "idt", "Irish (IDT)"),
        ("cy", "ccg", "Welsh (CCG)"),
        ("gd", "arcosg", "Scottish Gaelic (ARCOSG)"),
        ("br", "keb", "Breton (KEB)"),
        ("mt", "mudt", "Maltese (MUDT)"),
        ("is", "icepahc", "Icelandic (IcePaHC)"),
        ("fo", "farpahc", "Faroese (FarPaHC)"),
        ("nn", "nynorsk", "Norwegian Nynorsk"),
        ("be", "hse", "Belarusian (HSE)"),
        ("kk", "ktb", "Kazakh (KTB)"),
        ("hy", "armtdp", "Armenian (ArmTDP)"),
        ("mr", "ufal", "Marathi (UFAL)"),
        ("sa", "vedic", "Sanskrit (Vedic)"),
        ("la", "ittb", "Latin (ITTB)"),
        ("grc", "proiel", "Ancient Greek"),
        ("cu", "proiel", "Old Church Slavonic"),
        ("got", "proiel", "Gothic"),
        ("ug", "udt", "Uyghur (UDT)"),
        ("kk", "atk", "Kazakh (ATK)"),
        ("bn", "iitb", "Bengali (IITB)"),
        ("km", "wtb", "Khmer (WTB)"),
        ("lo", "pud", "Lao (PUD)"),
        ("my", "myu", "Myanmar (Myu)"),
    ]

    for lang, pkg, description in common_models:
        model_key = f"{lang}_{pkg}"
        entry = build_model_entry(
            "stanza",
            model_key,
            language_code=lang,
            language_name=clean_language_name(description),
            description=description,
            date=installed_models.get(model_key),
            preferred=pkg == "ewt" and lang == "en",
        )
        result[model_key] = entry

    if refresh_cache:
        try:
            write_model_cache_entry(cache_key, result)
        except (OSError, PermissionError):
            pass
    return result


def list_stanza_models(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
) -> int:
    try:
        entries = get_stanza_model_entries(
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        print(f"\nAvailable Stanza models:")
        print(f"{'Model ID':<20} {'ISO':<6} {'Language':<20} {'Status':<12}")
        print("=" * 65)
        for key in sorted(entries.keys()):
            entry = entries[key]
            iso = (entry.get(LANGUAGE_FIELD_ISO) or "")[:6]
            lang = entry.get(LANGUAGE_FIELD_NAME, "")
            status = entry.get("status", "")
            suffix = "*" if entry.get("preferred") else ""
            print(f"{key+suffix:<20} {iso:<6} {lang:<20} {status:<12}")
        preferred_note = "\n(*) Preferred model used by auto-selection"
        print(preferred_note)
        print(f"\nTotal: {len(entries)} model(s)")
        print("Stanza models are downloaded automatically on first use")
        print("Features: tokenization, lemma, upos, xpos, feats, depparse, NER")
        return 0
    except Exception as exc:
        print(f"Error listing Stanza models: {exc}")
        import traceback

        traceback.print_exc()
        return 1


class StanzaBackend(BackendManager):
    """Stanza-based neural backend."""

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
        enable_wsd: bool = False,
        enable_sentiment: bool = False,
        enable_coref: bool = False,
    ):
        setup_backend_environment("stanza")
        self._resources_dir = get_backend_models_dir("stanza", create=True)
        os.environ.setdefault("STANZA_RESOURCES_DIR", str(self._resources_dir))
        from stanza import Pipeline, download, resources

        self._Pipeline = Pipeline
        self._download = download
        self._resources = resources
        if not verbose:
            stanza_logger = logging.getLogger("stanza")
            stanza_logger.setLevel(logging.ERROR)
            for handler in stanza_logger.handlers:
                handler.setLevel(logging.ERROR)
            stanza_logger.propagate = False
        self._model_name = model_name
        base_language = language or (model_name.split("_")[0] if model_name and "_" in model_name else "en")
        self._language = base_language.lower()
        inferred_package = None
        if model_name and "_" in model_name:
            parts = model_name.split("_", 1)
            if len(parts) > 1:
                inferred_package = parts[1]
        self._package = (
            package
            or inferred_package
            or DEFAULT_STANZA_PACKAGES.get(self._language, "ewt")
        )
        default_processors = ["tokenize", "pos", "lemma", "depparse", "ner"]
        if enable_wsd:
            print("[flexipipe] Warning: Stanza WSD is not supported; ignoring --stanza-wsd.")
        extra_flags = {
            "sentiment": enable_sentiment,
            "coref": enable_coref,
        }
        if processors:
            self._processors = processors
        else:
            valid_extras = []
            for proc, flag in extra_flags.items():
                if not flag:
                    continue
                if proc == "wsd" and self._language not in {"en"}:
                    continue
                valid_extras.append(proc)
            self._processors = ",".join(default_processors + valid_extras)
        self._use_gpu = use_gpu
        self._download_model = download_model
        self._verbose = verbose
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline:
            return self._pipeline
        try:
            self._pipeline = self._Pipeline(
                lang=self._language,
                package=self._package,
                processors=self._processors,
                use_gpu=self._use_gpu,
                tokenize_pretokenized=False,
                dir=str(self._resources_dir),
            )
        except Exception:
            if self._download_model:
                self._download(
                    self._language,
                    package=self._package,
                    processors=self._processors,
                    resource_dir=str(self._resources_dir),
                )
                self._pipeline = self._Pipeline(
                    lang=self._language,
                    package=self._package,
                    processors=self._processors,
                    use_gpu=self._use_gpu,
                    tokenize_pretokenized=False,
                    dir=str(self._resources_dir),
                )
            else:
                raise
        return self._pipeline

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

        pipeline = self._ensure_pipeline()
        start_time = time.time()

        if use_raw_text or not document.sentences:
            text = "\n".join(sentence.text for sentence in document.sentences if sentence.text)
            stanza_doc = pipeline(text)
            result_doc = _stanza_doc_to_document(stanza_doc)
        else:
            pretokenized = [[token.form for token in sentence.tokens] for sentence in document.sentences]
            stanza_doc = pipeline(pretokenized=pretokenized)
            result_doc = _stanza_doc_to_document(stanza_doc, original_doc=document)

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
        raise NotImplementedError("Stanza training is not integrated into flexipipe.")

    def supports_training(self) -> bool:
        return False


def _stanza_doc_to_document(stanza_doc, original_doc: Optional[Document] = None) -> Document:
    doc = Document(id=original_doc.id if original_doc else "")
    if original_doc:
        doc.meta = original_doc.meta.copy()
        doc.attrs = original_doc.attrs.copy()

    for stanza_sent in stanza_doc.sentences:
        sent_id = getattr(stanza_sent, "sent_id", None) or ""
        sentence = Sentence(id=sent_id, sent_id=sent_id, text=stanza_sent.text, tokens=[])

        if hasattr(stanza_sent, "ents") and stanza_sent.ents:
            for ent in stanza_sent.ents:
                try:
                    start_idx = getattr(ent, "start", None)
                    end_idx = getattr(ent, "end", None)
                    if start_idx is None or end_idx is None:
                        continue
                    entity = Entity(
                        start=start_idx + 1,
                        end=end_idx,
                        label=getattr(ent, "type", ""),
                        text=getattr(ent, "text", ""),
                    )
                    sentence.entities.append(entity)
                except Exception:
                    continue

        for word in stanza_sent.words:
            token = Token(
                id=word.id if isinstance(word.id, int) else 0,
                form=word.text,
                lemma=word.lemma or "",
                upos=word.upos or "",
                xpos=word.xpos or "",
                feats=word.feats or "",
                head=word.head if word.head else 0,
                deprel=word.deprel or "",
                space_after=("SpaceAfter=No" not in (word.misc or "")) if word.misc else True,
            )
            sentence.tokens.append(token)

        if sentence.tokens:
            sentence.tokens[-1].space_after = None
        doc.sentences.append(sentence)

    return doc


def _create_stanza_backend(
    *,
    model_name: str | None = None,
    language: str | None = None,
    package: str | None = None,
    processors: str | None = None,
    use_gpu: bool = False,
    download_model: bool = False,
    verbose: bool = False,
    enable_wsd: bool = False,
    enable_sentiment: bool = False,
    enable_coref: bool = False,
    training: bool = False,
    **kwargs: Any,
) -> StanzaBackend:
    from ..backend_utils import validate_backend_kwargs
    
    validate_backend_kwargs(kwargs, "Stanza", allowed_extra=["training"])

    return StanzaBackend(
        model_name=model_name,
        language=language,
        package=package,
        processors=processors,
        use_gpu=use_gpu,
        download_model=download_model,
        verbose=verbose,
        enable_wsd=enable_wsd,
        enable_sentiment=enable_sentiment,
        enable_coref=enable_coref,
    )


BACKEND_SPEC = BackendSpec(
    name="stanza",
    description="Stanza - Stanford NLP library with high-quality models",
    factory=_create_stanza_backend,
    get_model_entries=get_stanza_model_entries,
    list_models=list_stanza_models,
    supports_training=False,
    is_rest=False,
    url="https://github.com/stanfordnlp/stanza",
)

