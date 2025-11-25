"""Flair backend implementation and registry spec."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..backend_spec import BackendSpec
from ..doc import Document, Entity, Sentence, SubToken, Token
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


def _document_to_plain_text(document: Document) -> str:
    parts: List[str] = []
    for sentence in document.sentences:
        if sentence.text:
            parts.append(sentence.text.strip())
        else:
            buf: List[str] = []
            for tok in sentence.tokens:
                buf.append(tok.form)
                if tok.space_after is not False:
                    buf.append(" ")
            parts.append("".join(buf).strip())
    return "\n".join(filter(None, parts))


def _flair_sentence_to_text(sentence) -> str:
    pieces: List[str] = []
    for token in sentence.tokens:
        pieces.append(token.text)
        if token.whitespace_after is not None:
            if isinstance(token.whitespace_after, str):
                pieces.append(token.whitespace_after)
            elif token.whitespace_after:
                pieces.append(" ")
        else:
            pieces.append(" ")
    return "".join(pieces).strip()


def get_flair_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Dict[str, str]]:
    cache_key = "flair"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            if verbose:
                print("[flexipipe] Using cached Flair model list (use --refresh-cache to update).")
            return cached

    if verbose:
        print("[flexipipe] Building Flair model list...")
    result: Dict[str, Dict[str, str]] = {}
    try:
        flair_cache = get_backend_models_dir("flair", create=False)
    except (OSError, PermissionError) as e:
        if verbose:
            print(f"[flexipipe] Warning: Could not access Flair models directory: {e}")
        return result
    installed_models: Dict[str, str] = {}
    if flair_cache.exists():
        from datetime import datetime
        try:
            for model_dir in flair_cache.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.pt"))
                    if model_files:
                        mtime = model_files[0].stat().st_mtime
                        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
                        installed_models[model_name] = date_str
        except (OSError, PermissionError) as e:
            if verbose:
                print(f"[flexipipe] Warning: Could not read Flair models directory: {e}")

    common_models = [
        ("upos-fast", "POS", "Universal POS tagging (fast)"),
        ("pos", "POS", "English POS tagging"),
        ("ner", "NER", "English NER"),
        ("ner-fast", "NER", "English NER (fast)"),
        ("chunk", "Chunking", "English chunking"),
        ("upos", "POS", "Universal POS tagging"),
        ("frame", "WSD", "Predicate sense tagging (PropBank / WSD)"),
        ("chunk-fast", "Chunking", "English chunking (fast)"),
        ("dependency-fast", "Dependency parsing", "Universal dependencies parser (fast)"),
        ("dependency", "Dependency parsing", "Universal dependencies parser"),
    ]

    for model, mtype, desc in common_models:
        status = "installed" if model in installed_models else "available"
        date = installed_models.get(model, "")
        entry = build_model_entry(
            "flair",
            model,
            language_code="en",
            language_name="English",
            type=mtype,
            description=desc,
            date=date,
        )
        entry["status"] = status
        result[model] = entry

    if refresh_cache:
        try:
            write_model_cache_entry(cache_key, result)
        except (OSError, PermissionError):
            pass
    return result


def list_flair_models(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> int:
    try:
        result = get_flair_model_entries(
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_ttl_seconds=MODEL_CACHE_TTL_SECONDS,
            verbose=True,
        )
        flair_cache = get_backend_models_dir("flair", create=False)

        print(f"\nAvailable Flair models:")
        print(f"{'Model Name':<40} {'Type':<20} {'Status':<25} {'Description':<50}")
        print("=" * 135)

        installed_count = 0
        for model_name in sorted(result.keys()):
            model_info = result[model_name]
            mtype = model_info.get("type", "")
            desc = model_info.get("description", "")
            status_str = model_info.get("status", "available")
            date = model_info.get("date", "")

            if status_str == "installed":
                status = f"✓ Installed ({date})" if date else "✓ Installed"
                installed_count += 1
            else:
                status = "Available"

            print(f"{model_name:<40} {mtype:<20} {status:<25} {desc:<50}")

        if installed_count > 0:
            print(f"\nNote: {installed_count} model(s) installed in {flair_cache}")
        from pathlib import Path
        default_flair_cache = Path.home() / ".flair" / "models"
        if default_flair_cache.exists() and default_flair_cache != flair_cache:
            default_count = len(list(default_flair_cache.glob("*.bin"))) + len(
                list(default_flair_cache.glob("*.pt"))
            )
            if default_count > 0:
                print(
                    f"Note: {default_count} model(s) found in {default_flair_cache} (not managed by flexipipe)"
                )
        print("\nNote: Flair models are not downloaded automatically.")
        print("Use --download-model with your flexipipe command to fetch missing models.")
        print("Features vary by model (POS models: UPOS/XPOS; NER models: named entities; WSD models: senses)")

        total_models = len(result)
        print(f"\nTotal: {total_models} model(s)")

        return 0
    except Exception as e:
        print(f"Error listing Flair models: {e}")
        import traceback

        traceback.print_exc()
        return 1


@dataclass
class _FlairModelSpec:
    upos: str
    xpos: Optional[str]
    ner: Optional[str]
    wsd: Optional[str]


DEFAULT_UPOS_MODELS = {"en": "upos"}
DEFAULT_XPOS_MODELS = {"en": "pos"}
DEFAULT_NER_MODELS = {"en": "ner"}
DEFAULT_WSD_MODELS = {"en": "frame"}


def _parse_model_spec(model_name: Optional[str], language: Optional[str]) -> _FlairModelSpec:
    if model_name and not language and len(model_name) <= 3 and model_name.isalpha():
        language = model_name
        model_name = None

    lang_key = (language or "").lower()
    spec = _FlairModelSpec(
        upos=DEFAULT_UPOS_MODELS.get(lang_key, "upos"),
        xpos=None,
        ner=DEFAULT_NER_MODELS.get(lang_key, "ner"),
        wsd=None,
    )

    if model_name:
        parts = [p.strip() for p in model_name.split(",") if p.strip()]
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ("upos", "pos"):
                    spec.upos = value or spec.upos
                elif key == "xpos":
                    spec.xpos = value or None
                elif key == "ner":
                    spec.ner = value or None
                elif key in ("wsd", "frame"):
                    spec.wsd = value or None
            else:
                spec.upos = part
    return spec


class FlairBackend(BackendManager):
    """Flair-based neural backend providing POS tagging and WSD."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        download_model: bool = False,
        verbose: bool = False,
    ):
        from ..model_storage import setup_backend_environment

        setup_backend_environment("flair")
        self._models_dir = get_backend_models_dir("flair", create=True)

        try:
            from flair.models import SequenceTagger
            from flair.data import Sentence as FlairSentence
        except ImportError as exc:
            raise ImportError(
                "Flair backend requires the 'flair' package. Install it with: pip install flair"
            ) from exc
        except (ValueError, RuntimeError) as exc:
            error_msg = str(exc).lower()
            error_str = str(exc)
            if "numpy" in error_msg and (
                "not available" in error_msg or "_ARRAY_API" in error_str or "dtype size changed" in error_msg
            ):
                try:
                    import numpy

                    numpy_version = getattr(numpy, "__version__", "unknown")
                    if numpy_version.startswith("2."):
                        raise ValueError(
                            f"NumPy version incompatibility detected: NumPy {numpy_version} is installed, "
                            "but PyTorch (required by Flair) was compiled against NumPy 1.x. "
                            "Downgrade NumPy with: pip install 'numpy<2'\nThen retry your command."
                        ) from exc
                except ImportError:
                    pass
            raise

    def _find_local_model_path(self, model_name: str) -> Optional[Path]:
        model_dir = self._models_dir / model_name
        if model_dir.is_dir():
            for pattern in ("*.bin", "*.pt"):
                files = list(model_dir.glob(pattern))
                if files:
                    return files[0]
        legacy_path = self._models_dir / f"{model_name}.pt"
        if legacy_path.exists():
            return legacy_path
        return None

    def _load_tagger(self, model_name: Optional[str]):
        if not model_name:
            return None
        try:
            local_path = self._find_local_model_path(model_name)
            if local_path:
                return self._SequenceTagger.load(str(local_path))
            if not self._download:
                raise ValueError(
                    f"Flair model '{model_name}' is not installed. "
                    "Re-run with --download-model to fetch Flair models automatically."
                )
            tagger = self._SequenceTagger.load(model_name)
            try:
                target_dir = self._models_dir / model_name
                target_dir.mkdir(parents=True, exist_ok=True)
                save_path = target_dir / f"{model_name}.pt"
                tagger.save(str(save_path))
            except Exception:
                pass
            return tagger
        except (ValueError, RuntimeError) as exc:
            error_msg = str(exc).lower()
            error_str = str(exc)
            if "numpy" in error_msg and (
                "not available" in error_msg or "_ARRAY_API" in error_str or "dtype size changed" in error_msg
            ):
                try:
                    import numpy

                    numpy_version = getattr(numpy, "__version__", "unknown")
                    if numpy_version.startswith("2."):
                        raise ValueError(
                            f"NumPy version incompatibility detected: NumPy {numpy_version} is installed, "
                            "but PyTorch (required by Flair) was compiled against NumPy 1.x. "
                            "Downgrade NumPy with: pip install 'numpy<2'\nThen retry your command."
                        ) from exc
                except ImportError:
                    pass
            if "repository not found" in error_msg or "401" in error_str or "not found" in error_msg:
                raise ValueError(
                    f"Flair model '{model_name}' not found or not accessible. "
                    f"Common valid model names include: 'ner', 'ner-fast', 'upos-fast', 'pos', 'chunk'. "
                    f"See Flair documentation for available models: https://github.com/flairNLP/flair"
                ) from exc
            raise

    def _update_model_descriptor(self) -> None:
        if self._descriptor_map:
            self.model_descriptor = ",".join(f"{key}={value}" for key, value in self._descriptor_map.items())
        else:
            self.model_descriptor = "unknown"

    def _default_xpos_model(self) -> Optional[str]:
        return DEFAULT_XPOS_MODELS.get(self._language)

    def _default_wsd_model(self) -> Optional[str]:
        return DEFAULT_WSD_MODELS.get(self._language)

    def _ensure_xpos_tagger(self) -> None:
        if self._xpos_tagger:
            return
        model_name = self._xpos_model_name or self._default_xpos_model()
        if not model_name:
            raise ValueError(
                "Flair backend does not have an XPOS model configured for this language. "
                "Provide one via --model xpos=<model_name> or use --download-model."
            )
        tagger = self._load_tagger(model_name)
        if tagger is None:
            raise ValueError(f"Failed to load Flair XPOS model '{model_name}'.")
        self._xpos_tagger = tagger
        self._xpos_tag_type = tagger.tag_type
        self._xpos_model_name = model_name
        self._descriptor_map["xpos"] = model_name
        self._update_model_descriptor()

    def _ensure_wsd_tagger(self) -> None:
        if self._wsd_tagger:
            return
        model_name = self._wsd_model_name or self._default_wsd_model()
        if not model_name:
            raise ValueError(
                "Flair backend does not have a WSD model configured for this language. "
                "Provide one via --model wsd=<model_name> (or frame=<model_name>) or use --download-model."
            )
        tagger = self._load_tagger(model_name)
        if tagger is None:
            raise ValueError(f"Failed to load Flair WSD model '{model_name}'.")
        self._wsd_tagger = tagger
        self._wsd_tag_type = tagger.tag_type
        self._wsd_model_name = model_name
        self._descriptor_map["wsd"] = model_name
        self._update_model_descriptor()

    def _build_flair_sentences(
        self,
        document: Document,
        use_raw_text: bool,
    ) -> Tuple[List, Optional[Document]]:
        if use_raw_text or not document.sentences:
            text = _document_to_plain_text(document)
            return [self._FlairSentence(text, language_code=self._language)], None
        flair_sentences = []
        for sent in document.sentences:
            if sent.text:
                text = sent.text.strip()
            else:
                parts = []
                for tok in sent.tokens:
                    parts.append(tok.form)
                    if tok.space_after is not False:
                        parts.append(" ")
                text = "".join(parts).strip()
            if text:
                flair_sentences.append(self._FlairSentence(text, language_code=self._language))
        return flair_sentences, document

    def _flair_to_document(
        self,
        flair_sentences,
        original_doc: Optional[Document],
        need_upos: bool,
        need_xpos: bool,
        need_wsd: bool,
    ) -> Document:
        doc_id = original_doc.id if original_doc else ""
        doc_meta = dict(original_doc.meta) if original_doc else {}
        result = Document(id=doc_id, meta=doc_meta)

        tokid_to_original: dict[str, Token] = {}
        if original_doc:
            for sent in original_doc.sentences:
                for tok in sent.tokens:
                    if tok.tokid:
                        tokid_to_original[tok.tokid] = tok
                    if tok.is_mwt and tok.subtokens:
                        for sub in tok.subtokens:
                            if sub.tokid:
                                tokid_to_original[sub.tokid] = tok

        upos_tag_type = self._upos_tagger.tag_type if (need_upos and self._upos_tagger) else None
        xpos_tag_type = self._xpos_tagger.tag_type if (need_xpos and self._xpos_tagger) else None

        def _apply_wsd_label(flair_token, target_token: Token) -> None:
            if not need_wsd or not self._wsd_tag_type:
                return
            try:
                label = flair_token.get_label(self._wsd_tag_type)
            except (AttributeError, KeyError):
                return
            if not label:
                return
            value = getattr(label, "value", "") or ""
            if not value or value.upper() == "O":
                return
            target_token.attrs["wsd"] = value
            score = getattr(label, "score", None)
            if score is not None:
                target_token.attrs["wsd_confidence"] = float(score)

        token_offset = 0
        for sent_idx, flair_sentence in enumerate(flair_sentences):
            orig_sentence = (
                original_doc.sentences[sent_idx]
                if original_doc and sent_idx < len(original_doc.sentences)
                else None
            )

            flair_tokens_with_tokid: List[Tuple[Token, Optional[str]]] = []

            if orig_sentence and orig_sentence.tokens:
                orig_forms = [tok.form.lower() for tok in orig_sentence.tokens]
                flair_forms = [tok.text.lower() for tok in flair_sentence.tokens]
                matcher = SequenceMatcher(None, orig_forms, flair_forms, autojunk=False)
                alignment: List[Optional[int]] = [None] * len(flair_sentence.tokens)
                used_orig_indices = set()

                for block in matcher.get_matching_blocks():
                    orig_start, flair_start, size = block
                    for i in range(size):
                        orig_idx = orig_start + i
                        flair_idx = flair_start + i
                        if flair_idx < len(alignment) and orig_idx not in used_orig_indices:
                            alignment[flair_idx] = orig_idx
                            used_orig_indices.add(orig_idx)

                for flair_idx, flair_token in enumerate(flair_sentence.tokens):
                    if alignment[flair_idx] is None:
                        best_orig_idx = None
                        best_score = 0.0
                        for orig_idx, orig_tok in enumerate(orig_sentence.tokens):
                            if orig_idx in used_orig_indices:
                                continue
                            if flair_token.text.lower() == orig_tok.form.lower():
                                best_orig_idx = orig_idx
                                best_score = 1.0
                                break
                            elif flair_token.text.lower() in orig_tok.form.lower() or orig_tok.form.lower() in flair_token.text.lower():
                                score = min(len(flair_token.text), len(orig_tok.form)) / max(len(flair_token.text), len(orig_tok.form))
                                if score > best_score:
                                    best_score = score
                                    best_orig_idx = orig_idx
                        if best_orig_idx is not None:
                            alignment[flair_idx] = best_orig_idx
                            used_orig_indices.add(best_orig_idx)

                for flair_idx, flair_token in enumerate(flair_sentence.tokens):
                    orig_idx = alignment[flair_idx] if flair_idx < len(alignment) else None
                    orig_tok = (
                        orig_sentence.tokens[orig_idx]
                        if orig_idx is not None and orig_idx < len(orig_sentence.tokens)
                        else None
                    )

                    upos = ""
                    upos_confidence = None
                    xpos = ""
                    xpos_confidence = None
                    if need_upos and self._upos_tagger and upos_tag_type:
                        try:
                            label = flair_token.get_label(upos_tag_type)
                            if label:
                                upos = label.value
                                upos_confidence = float(label.score) if hasattr(label, "score") else None
                        except (AttributeError, KeyError):
                            pass
                    if need_xpos and self._xpos_tagger and xpos_tag_type:
                        try:
                            label = flair_token.get_label(xpos_tag_type)
                            if label:
                                xpos = label.value
                                xpos_confidence = float(label.score) if hasattr(label, "score") else None
                        except (AttributeError, KeyError):
                            pass

                    assigned_tokid = None
                    if orig_tok and orig_tok.tokid:
                        assigned_tokid = orig_tok.tokid
                    elif orig_tok and orig_tok.is_mwt and orig_tok.subtokens:
                        assigned_tokid = orig_tok.tokid

                    if orig_tok:
                        space_after = orig_tok.space_after
                    elif flair_token.whitespace_after is not None:
                        space_after = bool(flair_token.whitespace_after)
                    else:
                        space_after = True

                    token = Token(
                        id=orig_tok.id if orig_tok else flair_idx + 1,
                        form=orig_tok.form if orig_tok else flair_token.text,
                        lemma="",
                        upos=upos,
                        xpos=xpos,
                        feats="",
                        head=0,
                        deprel="",
                        space_after=space_after,
                        tokid=assigned_tokid or "",
                        upos_confidence=upos_confidence,
                        xpos_confidence=xpos_confidence,
                    )
                    _apply_wsd_label(flair_token, token)
                    flair_tokens_with_tokid.append((token, assigned_tokid))
            else:
                for tok_idx, flair_token in enumerate(flair_sentence.tokens):
                    upos = ""
                    upos_confidence = None
                    xpos = ""
                    xpos_confidence = None
                    if need_upos and self._upos_tagger and upos_tag_type:
                        try:
                            label = flair_token.get_label(upos_tag_type)
                            if label:
                                upos = label.value
                                upos_confidence = float(label.score) if hasattr(label, "score") else None
                        except (AttributeError, KeyError):
                            pass
                    if need_xpos and self._xpos_tagger and xpos_tag_type:
                        try:
                            label = flair_token.get_label(xpos_tag_type)
                            if label:
                                xpos = label.value
                                xpos_confidence = float(label.score) if hasattr(label, "score") else None
                        except (AttributeError, KeyError):
                            pass
                    token = Token(
                        id=tok_idx + 1,
                        form=flair_token.text,
                        lemma="",
                        upos=upos,
                        xpos=xpos,
                        feats="",
                        head=0,
                        deprel="",
                        space_after=bool(flair_token.whitespace_after) if flair_token.whitespace_after is not None else True,
                        tokid=f"s{sent_idx + 1}-t{tok_idx + 1}",
                        upos_confidence=upos_confidence,
                        xpos_confidence=xpos_confidence,
                    )
                    _apply_wsd_label(flair_token, token)
                    flair_tokens_with_tokid.append((token, token.tokid))

            tokens: List[Token] = []
            pending_mwt_tokens: List[Token] = []
            pending_mwt_tokid: Optional[str] = None

            for token, tokid in flair_tokens_with_tokid:
                should_merge = False
                if tokid and pending_mwt_tokid is not None and tokid == pending_mwt_tokid:
                    orig_tok_for_mwt = tokid_to_original.get(tokid) if tokid else None
                    if orig_tok_for_mwt and orig_tok_for_mwt.is_mwt:
                        should_merge = True
                    elif len(pending_mwt_tokens) > 0:
                        should_merge = True

                if should_merge:
                    pending_mwt_tokens.append(token)
                else:
                    if pending_mwt_tokens:
                        if len(pending_mwt_tokens) > 1:
                            orig_tok_for_mwt = tokid_to_original.get(pending_mwt_tokid) if pending_mwt_tokid else None
                            mwt_form = (
                                orig_tok_for_mwt.form
                                if orig_tok_for_mwt and orig_tok_for_mwt.form
                                else "".join(t.form for t in pending_mwt_tokens)
                            )

                            subtokens = []
                            for idx, sub_token in enumerate(pending_mwt_tokens):
                                subtoken = SubToken(
                                    id=sub_token.id,
                                    form=sub_token.form,
                                    lemma=sub_token.lemma,
                                    upos=sub_token.upos,
                                    xpos=sub_token.xpos,
                                    feats=sub_token.feats,
                                    space_after=(idx < len(pending_mwt_tokens) - 1)
                                    or pending_mwt_tokens[-1].space_after,
                                    tokid=pending_mwt_tokid or "",
                                    upos_confidence=sub_token.upos_confidence,
                                )
                                subtokens.append(subtoken)

                            mwt_token = Token(
                                id=pending_mwt_tokens[0].id,
                                form=mwt_form,
                                lemma=pending_mwt_tokens[0].lemma,
                                upos=pending_mwt_tokens[0].upos,
                                xpos=pending_mwt_tokens[0].xpos,
                                feats=pending_mwt_tokens[0].feats,
                                head=pending_mwt_tokens[0].head,
                                deprel=pending_mwt_tokens[0].deprel,
                                space_after=pending_mwt_tokens[-1].space_after,
                                tokid=pending_mwt_tokid or "",
                                is_mwt=True,
                                mwt_start=pending_mwt_tokens[0].id,
                                mwt_end=pending_mwt_tokens[-1].id,
                                subtokens=subtokens,
                                upos_confidence=pending_mwt_tokens[0].upos_confidence,
                            )
                            tokens.append(mwt_token)
                        else:
                            tokens.append(pending_mwt_tokens[0])
                        pending_mwt_tokens = []
                        pending_mwt_tokid = None

                    if tokid:
                        pending_mwt_tokens = [token]
                        pending_mwt_tokid = tokid
                    else:
                        tokens.append(token)

            if pending_mwt_tokens:
                if len(pending_mwt_tokens) > 1:
                    orig_tok_for_mwt = tokid_to_original.get(pending_mwt_tokid) if pending_mwt_tokid else None
                    mwt_form = (
                        orig_tok_for_mwt.form
                        if orig_tok_for_mwt and orig_tok_for_mwt.form
                        else "".join(t.form for t in pending_mwt_tokens)
                    )

                    subtokens = []
                    for idx, sub_token in enumerate(pending_mwt_tokens):
                        subtoken = SubToken(
                            id=sub_token.id,
                            form=sub_token.form,
                            lemma=sub_token.lemma,
                            upos=sub_token.upos,
                            xpos=sub_token.xpos,
                            feats=sub_token.feats,
                            space_after=(idx < len(pending_mwt_tokens) - 1) or pending_mwt_tokens[-1].space_after,
                            tokid=pending_mwt_tokid or "",
                            upos_confidence=sub_token.upos_confidence,
                        )
                        subtokens.append(subtoken)

                    mwt_token = Token(
                        id=pending_mwt_tokens[0].id,
                        form=mwt_form,
                        lemma=pending_mwt_tokens[0].lemma,
                        upos=pending_mwt_tokens[0].upos,
                        xpos=pending_mwt_tokens[0].xpos,
                        feats=pending_mwt_tokens[0].feats,
                        head=pending_mwt_tokens[0].head,
                        deprel=pending_mwt_tokens[0].deprel,
                        space_after=pending_mwt_tokens[-1].space_after,
                        tokid=pending_mwt_tokid or "",
                        is_mwt=True,
                        mwt_start=pending_mwt_tokens[0].id,
                        mwt_end=pending_mwt_tokens[-1].id,
                        subtokens=subtokens,
                        upos_confidence=pending_mwt_tokens[0].upos_confidence,
                    )
                    tokens.append(mwt_token)
                else:
                    tokens.append(pending_mwt_tokens[0])

            from ..conllu import _create_implicit_mwt

            temp_sentence = Sentence(
                id=orig_sentence.id if orig_sentence else f"s{sent_idx + 1}",
                sent_id=orig_sentence.sent_id if orig_sentence else f"s{sent_idx + 1}",
                text=orig_sentence.text if orig_sentence and orig_sentence.text else _flair_sentence_to_text(flair_sentence),
                tokens=tokens,
            )
            temp_sentence = _create_implicit_mwt(temp_sentence)
            tokens = temp_sentence.tokens

            entities: List[Entity] = []
            if self._ner_tagger:
                for span in flair_sentence.get_spans(self._ner_tag_type):
                    if not span.tokens:
                        continue
                    start = span.tokens[0].idx
                    end = span.tokens[-1].idx
                    entities.append(
                        Entity(
                            start=start,
                            end=end,
                            label=span.tag,
                            text=span.text,
                            attrs={},
                        )
                    )

            sentence = Sentence(
                id=orig_sentence.id if orig_sentence else f"s{sent_idx + 1}",
                sent_id=orig_sentence.sent_id if orig_sentence else f"s{sent_idx + 1}",
                text=orig_sentence.text if orig_sentence and orig_sentence.text else _flair_sentence_to_text(flair_sentence),
                tokens=tokens,
                entities=entities,
            )
            result.sentences.append(sentence)
            token_offset += len(sentence.tokens)
        return result

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        del overrides, preserve_pos_tags

        flair_sentences, original_doc = self._build_flair_sentences(document, use_raw_text)

        component_set = set(components) if components else None
        if component_set is None:
            need_upos = True
            need_xpos = False
            need_ner = self._ner_tagger is not None
            need_wsd = False
        else:
            need_upos = "upos" in component_set or "tag" in component_set or "parse" in component_set
            need_xpos = "xpos" in component_set
            need_ner = "ner" in component_set
            need_wsd = "wsd" in component_set

        start_time = time.time()

        if need_upos:
            if not self._upos_tagger:
                raise ValueError("Flair UPOS model is not available.")
            self._upos_tagger.predict(flair_sentences, verbose=False)

        if need_xpos:
            self._ensure_xpos_tagger()
            if not self._xpos_tagger:
                raise ValueError("Flair XPOS model is not available.")
            self._xpos_tagger.predict(flair_sentences, verbose=False)

        if need_ner:
            if not self._ner_tagger:
                raise ValueError("Flair NER model is not available.")
            self._ner_tagger.predict(flair_sentences, verbose=False)

        if need_wsd:
            self._ensure_wsd_tagger()
            if not self._wsd_tagger:
                raise ValueError("Flair WSD model is not available.")
            self._wsd_tagger.predict(flair_sentences, verbose=False)

        elapsed = time.time() - start_time

        result_doc = self._flair_to_document(
            flair_sentences,
            original_doc,
            need_upos,
            need_xpos,
            need_wsd,
        )
        token_count = sum(len(sent.tokens) for sent in result_doc.sentences)
        stats = {
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
            "sentences_per_second": len(result_doc.sentences) / elapsed if elapsed > 0 else 0.0,
        }
        return NeuralResult(document=result_doc, stats=stats)

    def train(self, *args, **kwargs):  # pragma: no cover - not implemented
        raise NotImplementedError("Flair backend training is not implemented.")

    def supports_training(self) -> bool:  # pragma: no cover - trivial
        return False


def _create_flair_backend(
    *,
    model_name: str | None = None,
    language: str | None = None,
    download_model: bool = False,
    verbose: bool = False,
    training: bool = False,
    **kwargs: Any,
) -> FlairBackend:
    from ..backend_utils import validate_backend_kwargs
    
    validate_backend_kwargs(kwargs, "Flair", allowed_extra=["training"])
    return FlairBackend(
        model_name=model_name,
        language=language,
        download_model=download_model,
        verbose=verbose,
    )


BACKEND_SPEC = BackendSpec(
    name="flair",
    description="Flair - State-of-the-art sequence tagging library",
    factory=_create_flair_backend,
    get_model_entries=get_flair_model_entries,
    list_models=list_flair_models,
    supports_training=False,
    is_rest=False,
)
