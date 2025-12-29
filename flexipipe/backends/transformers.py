"""HuggingFace Transformers backend spec and implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..backend_spec import BackendSpec
from ..doc import Document, Span
from ..model_registry import merge_remote_and_local_models
from ..model_storage import get_backend_models_dir, setup_backend_environment
from ..neural_backend import BackendManager, NeuralResult


SUPPORTED_TRANSFORMER_TASKS = {"tag", "ner"}
MIN_TORCH_VERSION = (2, 6, 0)
MIN_TORCH_VERSION_STR = "2.6.0"
TORCH_CVE_URL = "https://nvd.nist.gov/vuln/detail/CVE-2025-32434"


def _clear_entity_misc(token) -> None:
    if not token.misc:
        return
    entries = [entry for entry in token.misc.split("|") if not entry.startswith("Entity=")]
    token.misc = "|".join(entries) if entries else ""


def _set_entity_misc(token, tag: Optional[str]) -> None:
    entries = [entry for entry in (token.misc.split("|") if token.misc else []) if not entry.startswith("Entity=")]
    if tag:
        entries.append(f"Entity={tag}")
    token.misc = "|".join(entries) if entries else ""


def _normalize_task(task: Optional[str]) -> Optional[str]:
    if not task:
        return None
    lowered = task.lower()
    if lowered in {"tag", "upos", "pos"}:
        return "tag"
    if lowered in {"ner", "entity", "entities", "named"}:
        return "ner"
    return lowered


def _parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    """Convert version strings like '2.6.0+cpu' into a tuple of integers."""
    if not version_str:
        return ()
    clean = version_str.split("+", 1)[0].split("-", 1)[0]
    parts: List[int] = []
    for chunk in clean.split("."):
        if not chunk:
            continue
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_is_at_least(current: Tuple[int, ...], minimum: Tuple[int, ...]) -> bool:
    length = max(len(current), len(minimum))
    for idx in range(length):
        cur_val = current[idx] if idx < len(current) else 0
        min_val = minimum[idx] if idx < len(minimum) else 0
        if cur_val < min_val:
            return False
        if cur_val > min_val:
            return True
    return True


def _ensure_secure_torch_version(torch_module) -> None:
    """Ensure torch is patched against CVE-2025-32434."""
    version_str = getattr(torch_module, "__version__", "0")
    version_tuple = _parse_version_tuple(version_str)
    if not _version_is_at_least(version_tuple, MIN_TORCH_VERSION):
        raise RuntimeError(
            "[flexipipe] Due to CVE-2025-32434 in torch.load, transformers backend now requires "
            f"torch>={MIN_TORCH_VERSION_STR}. Installed version is {version_str!r}. "
            "Upgrade via: pip install --upgrade 'torch>=2.6'. "
            f"See {TORCH_CVE_URL}"
        )


class HuggingFaceTransformersBackend(BackendManager):
    """Runtime backend that performs token classification using HuggingFace models."""

    def __init__(
        self,
        *,
        model_name: str,
        adapter_name: Optional[str] = None,
        device: str = "cpu",
        task: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        context_attrs: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        cache_dir = get_backend_models_dir("transformers", create=True)
        setup_backend_environment("transformers")

        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForTokenClassification,
                AutoTokenizer,
                TokenClassificationPipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "Transformers backend requires 'torch' and 'transformers'. "
                "Install them with: pip install torch transformers"
            ) from exc

        _ensure_secure_torch_version(torch)

        self._torch = torch
        self._model_name = model_name
        self._adapter_name = adapter_name
        self._task = _normalize_task(task)
        self._context_attrs = [attr.strip() for attr in (context_attrs or []) if attr and attr.strip()]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=str(cache_dir),
            use_fast=True,
        )
        if not tokenizer.is_fast:
            raise ValueError(
                f"Transformers backend requires a fast tokenizer. Model '{model_name}' does not expose word alignment metadata."
            )

        config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=str(cache_dir),
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=str(cache_dir),
        )

        self._device = self._prepare_device(device)
        model.to(self._device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._id2label = config.id2label or {idx: label for label, idx in config.label2id.items()}
        
        # Create pipeline for raw text mode with aggregation strategy
        # This handles subword grouping much better than manual grouping
        # Suppress "Device set to use cpu" message from transformers library unless verbose
        if not verbose:
            import sys
            import io
            old_stderr = sys.stderr
            try:
                sys.stderr = io.StringIO()
                self._pipeline = TokenClassificationPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",  # Groups subwords into words
                    device=self._device,
                )
            finally:
                sys.stderr = old_stderr
        else:
            self._pipeline = TokenClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",  # Groups subwords into words
                device=self._device,
            )

        if self._task is None:
            self._task = self._infer_task_from_labels()
        if self._task not in SUPPORTED_TRANSFORMER_TASKS:
            raise ValueError(
                f"Transformers backend currently supports {sorted(SUPPORTED_TRANSFORMER_TASKS)} tasks. "
                f"Model '{model_name}' appears to target '{self._task or 'unknown'}'. "
                "Specify --transformers-task or choose a token-classification model."
            )

    @property
    def supports_training(self) -> bool:
        return False

    def _prepare_device(self, requested: str) -> "torch.device":
        torch = self._torch
        req = (requested or "cpu").lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if req in {"mps", "apple"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _infer_task_from_labels(self) -> str:
        labels = {str(label).upper() for label in self._id2label.values()}
        if any(label.startswith(("B-", "I-", "E-", "S-")) for label in labels):
            return "ner"
        return "tag"

    def _prepare_token_input(self, token) -> str:
        base = token.form or token.attrs.get("text") or ""
        if not self._context_attrs:
            return base
        parts: List[str] = []
        for attr in self._context_attrs:
            value = self._extract_context_value(token, attr)
            if value:
                parts.append(f"{attr}={value}")
        if not parts:
            return base
        return f"{base} [{' '.join(parts)}]"

    def _extract_context_value(self, token, attr: str) -> Optional[str]:
        key = attr.lower()
        if key == "upos":
            return token.upos or None
        if key == "xpos":
            return token.xpos or None
        if key == "lemma":
            return token.lemma or None
        if key == "feats":
            return token.feats or None
        if key.startswith("attr:"):
            return token.attrs.get(key.split(":", 1)[1])
        return token.attrs.get(key)

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        if document is None:
            raise ValueError("Document cannot be None")
        total_tokens = 0
        
        if use_raw_text:
            # Raw text mode: use sentence text and let tokenizer handle tokenization
            # This allows the model to properly tokenize into subwords as it was trained
            for sentence in document.sentences:
                if not sentence.text:
                    continue
                text = sentence.text.strip()
                if not text:
                    continue
                try:
                    labels, confidences, word_tokens = self._predict_labels_from_raw_text(text)
                    # Create new tokens from tokenizer output, preserving original text
                    new_tokens = self._create_tokens_from_raw_text(sentence, labels, confidences, word_tokens, text)
                    sentence.tokens = new_tokens
                    total_tokens += len(new_tokens)
                except (ValueError, TypeError) as e:
                    # If raw text mode fails (e.g., no offset mapping support), fall back to pre-tokenized
                    if not sentence.tokens:
                        # If no tokens exist, create from text by splitting
                        words = text.split()
                        labels, confidences = self._predict_labels(words)
                        total_tokens += len(labels)
                        # Create tokens from split words
                        from ..doc import Token
                        new_tokens = []
                        for idx, (word, label, score) in enumerate(zip(words, labels, confidences)):
                            token = Token(
                                id=idx + 1,
                                form=word,
                                upos=label if self._task == "tag" else "",
                                upos_confidence=score,
                            )
                            new_tokens.append(token)
                        sentence.tokens = new_tokens
                    else:
                        # Use existing tokens
                        words = [self._prepare_token_input(token) for token in sentence.tokens]
                        labels, confidences = self._predict_labels(words)
                        total_tokens += len(labels)
                        if self._task == "ner":
                            self._apply_ner_labels(sentence, labels, confidences)
                        else:
                            self._apply_pos_labels(sentence, labels, confidences)
        else:
            # Pre-tokenized mode: use existing tokens
            for sentence in document.sentences:
                if not sentence.tokens:
                    continue
                words = [self._prepare_token_input(token) for token in sentence.tokens]
                labels, confidences = self._predict_labels(words)
                total_tokens += len(labels)
                if self._task == "ner":
                    self._apply_ner_labels(sentence, labels, confidences)
                else:
                    self._apply_pos_labels(sentence, labels, confidences)
        
        stats = {
            "model": self._model_name,
            "task": self._task,
            "tokens": total_tokens,
        }
        return NeuralResult(document=document, stats=stats)

    def _predict_labels(self, words: List[str]) -> tuple[List[str], List[float]]:
        tokenizer = self._tokenizer
        torch = self._torch
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        word_ids = encoding.word_ids(batch_index=0)
        encoding = {key: value.to(self._device) for key, value in encoding.items()}
        with torch.no_grad():
            logits = self._model(**encoding).logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_scores, pred_ids = torch.max(probs, dim=-1)
        labels = [""] * len(words)
        confidences = [0.0] * len(words)
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id >= len(words):
                continue
            if labels[word_id]:
                continue
            label_id = pred_ids[idx].item()
            labels[word_id] = self._id2label.get(label_id, str(label_id))
            confidences[word_id] = float(pred_scores[idx].item())
        default_label = "O" if self._task == "ner" else "X"
        for i in range(len(labels)):
            if not labels[i]:
                labels[i] = default_label
        return labels, confidences

    def _predict_labels_from_raw_text(self, text: str) -> tuple[List[str], List[float], List[dict]]:
        """
        Predict labels from raw text using TokenClassificationPipeline with aggregation.
        
        This uses HuggingFace's built-in aggregation strategy which handles subword
        grouping much better than manual grouping.
        """
        # Use the pipeline which handles subword aggregation automatically
        outputs = self._pipeline(text)
        
        # Convert pipeline output to our format
        labels: List[str] = []
        confidences: List[float] = []
        word_tokens: List[dict] = []
        
        for entity in outputs:
            # Extract the label - for POS tagging, it's in 'entity_group'
            # For NER, it's also in 'entity_group' but might have B-/I- prefixes
            label = entity.get("entity_group", "")
            if not label:
                # Fallback to 'label' field if 'entity_group' is not present
                label = entity.get("label", "")
            
            # The label should already be in the correct format from the pipeline
            # but we can validate it against our id2label mapping
            if label and label not in self._id2label.values():
                # Try to find a matching label (case-insensitive)
                label_upper = label.upper()
                matching_label = None
                for known_label in self._id2label.values():
                    if str(known_label).upper() == label_upper:
                        matching_label = str(known_label)
                        break
                if matching_label:
                    label = matching_label
                elif not label:
                    label = "X"  # Default label
            
            score = entity.get("score", 0.0)
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            
            # Extract word text from original text using offsets to preserve exact text
            word_text = text[start:end] if start < len(text) and end <= len(text) else entity.get("word", "")
            
            if not word_text.strip():
                continue  # Skip empty tokens
            
            labels.append(label)
            confidences.append(float(score))
            word_tokens.append({
                "start": start,
                "end": end,
                "label": label,
                "confidence": float(score),
            })
        
        default_label = "O" if self._task == "ner" else "X"
        for i in range(len(labels)):
            if not labels[i]:
                labels[i] = default_label
        
        return labels, confidences, word_tokens
    
    def _create_tokens_from_raw_text(
        self,
        sentence,
        labels: List[str],
        confidences: List[float],
        word_tokens: List[dict],
        original_text: str,
    ) -> List:
        """Create Token objects from word token info, preserving original text exactly."""
        from ..doc import Token
        
        tokens = []
        for idx, (label, confidence, word_info) in enumerate(zip(labels, confidences, word_tokens)):
            start = word_info["start"]
            end = word_info["end"]
            
            # Extract token text from original text using offsets - this preserves the exact text
            if start < len(original_text) and end <= len(original_text):
                token_text = original_text[start:end]
            else:
                continue  # Skip invalid offsets
            
            if not token_text.strip():
                continue  # Skip whitespace-only tokens
            
            token = Token(
                id=idx + 1,
                form=token_text,
                upos=label if self._task == "tag" else "",
                upos_confidence=confidence,
            )
            tokens.append(token)
        
        # Set space_after based on actual text between tokens
        for i in range(len(tokens) - 1):
            if i + 1 < len(word_tokens):
                current_end = word_tokens[i]["end"]
                next_start = word_tokens[i + 1]["start"]
                # Check if there's whitespace between tokens
                if current_end < len(original_text) and next_start <= len(original_text):
                    between = original_text[current_end:next_start]
                    tokens[i].space_after = bool(between.strip() == "" and between != "")
                else:
                    tokens[i].space_after = True
        if tokens:
            tokens[-1].space_after = None
        
        return tokens
    
    def _apply_pos_labels(self, sentence, labels: List[str], confidences: List[float]) -> None:
        for token, label, score in zip(sentence.tokens, labels, confidences):
            token.upos = label
            token.upos_confidence = score

    def _apply_ner_labels(self, sentence, labels: List[str], confidences: List[float]) -> None:
        ner_spans = sentence.spans.setdefault("ner", [])
        current_label = None
        current_start = None
        current_scores: List[float] = []
        for token in sentence.tokens:
            _clear_entity_misc(token)
        def flush(end_index: int) -> None:
            nonlocal current_label, current_start, current_scores
            if current_label is None or current_start is None:
                return
            avg_score = sum(current_scores) / len(current_scores) if current_scores else None
            span = Span(label=current_label, start=current_start, end=end_index)
            if avg_score is not None:
                span.attrs["confidence"] = avg_score
            ner_spans.append(span)
            current_label = None
            current_start = None
            current_scores = []
        tokens = sentence.tokens
        for idx, (label, score) in enumerate(zip(labels, confidences)):
            tag = label or "O"
            if "-" in tag:
                prefix, entity = tag.split("-", 1)
            else:
                prefix, entity = tag, ""
            prefix = prefix.upper()
            entity_clean = entity.upper() if entity else entity
            if prefix in {"B", "S"}:
                flush(idx)
                current_label = entity
                current_start = idx
                current_scores = [score]
                _set_entity_misc(tokens[idx], f"B-{entity_clean or entity}")
                if prefix == "S":
                    flush(idx + 1)
            elif prefix == "I" and current_label == entity:
                current_scores.append(score)
                _set_entity_misc(tokens[idx], f"I-{entity_clean or entity}")
            elif prefix == "E" and current_label == entity:
                current_scores.append(score)
                flush(idx + 1)
                _set_entity_misc(tokens[idx], f"I-{entity_clean or entity}")
            else:
                _set_entity_misc(tokens[idx], None)
                flush(idx)
        flush(len(labels))

    def train(
        self,
        train_data: Document | List[Document] | Path,
        output_dir: Path,
        *,
        dev_data: Document | List[Document] | Path | None = None,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError("Transformers backend training will be added in a future update.")


def _resolve_transformers_model_name(
    model_name: Optional[str],
    language: Optional[str],
) -> Optional[str]:
    """Resolve transformers model name using central resolution function."""
    from ..backend_utils import resolve_model_from_language
    
    return resolve_model_from_language(
        language=language,
        backend_name="transformers",
        model_name=model_name,
        preferred_only=True,
        use_cache=True,
    )


def _create_transformers_backend(
    *,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    task: Optional[str] = None,
    adapter_name: Optional[str] = None,
    device: str = "cpu",
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    context_attrs: Optional[List[str]] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> HuggingFaceTransformersBackend:
    # Resolve model name from language if not provided (using central function)
    from ..backend_utils import resolve_model_from_language
    
    try:
        resolved_model_name = resolve_model_from_language(
            language=language,
            backend_name="transformers",
            model_name=model_name,
            preferred_only=True,
            use_cache=True,
        )
    except ValueError as e:
        # Re-raise with context if language was provided but no model found
        if language:
            raise
        # If no language provided, provide a clearer error
        raise ValueError("Transformers backend requires --model to specify a HuggingFace model name, or --language to auto-select a model.") from e
    
    if not resolved_model_name:
        if language:
            raise ValueError(
                f"[flexipipe] No transformers model found for language '{language}'. "
                f"Use --model to specify a HuggingFace model name, or run 'flexipipe info models --backend transformers --language {language}' to see available models."
            )
        else:
            raise ValueError("Transformers backend requires --model to specify a HuggingFace model name, or --language to auto-select a model.")
    
    unexpected = set(kwargs) - {"download_model", "training"}
    if unexpected:
        raise ValueError(f"Unexpected Transformers backend arguments: {', '.join(sorted(unexpected))}")
    normalized_task = _normalize_task(task)
    
    return HuggingFaceTransformersBackend(
        model_name=resolved_model_name,
        adapter_name=adapter_name,
        device=device,
        task=normalized_task,
        revision=revision,
        trust_remote_code=trust_remote_code,
        context_attrs=context_attrs,
        verbose=verbose,
    )


def _infer_tasks_from_labels(labels: List[str]) -> List[str]:
    if not labels:
        return []
    upper = [label.upper() for label in labels]
    if any(label.startswith(("B-", "I-", "E-", "S-")) for label in upper):
        return ["ner"]
    return ["tag"]


def _load_local_transformers_metadata(model_dir: Path, model_name: Optional[str] = None) -> tuple[List[str], Optional[str]]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return [], None
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception:
        return [], None
    id2label = config.get("id2label")
    if isinstance(id2label, dict):
        labels = list(id2label.values())
    elif isinstance(id2label, list):
        labels = id2label
    else:
        labels = []
    tasks = _infer_tasks_from_labels(labels)
    language = config.get("language") or config.get("lang")
    
    # Check if language indicates multilingual
    if isinstance(language, str) and language.lower() in {"multi", "multilingual"}:
        language = "xx"
    # Also check model name for multilingual indicators
    elif model_name and isinstance(model_name, str):
        model_name_lower = model_name.lower()
        if any(indicator in model_name_lower for indicator in ["multilingual", "multi-lingual", "xlm", "mbert", "m-bert"]):
            language = "xx"
    
    return tasks, language


def _convert_hf_cache_name_to_model_name(cache_name: str) -> str:
    """Convert HuggingFace cache directory name to model name.
    
    Examples:
        models--Davlan--bert-base-multilingual-cased-ner-hrl -> Davlan/bert-base-multilingual-cased-ner-hrl
        models--bert-base-uncased -> bert-base-uncased
    """
    if cache_name.startswith("models--"):
        # Remove "models--" prefix and replace "--" with "/"
        model_name = cache_name[8:].replace("--", "/")
        return model_name
    return cache_name


def _find_latest_snapshot(model_cache_dir: Path) -> Optional[Path]:
    """Find the latest snapshot directory in a HuggingFace model cache directory."""
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    
    # Get all snapshot directories (they're named with commit hashes)
    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshots:
        return None
    
    # Return the first one (they're typically all the same, but we want the latest)
    # We could sort by modification time, but for now just return the first
    return snapshots[0]


def _build_transformers_model_entries() -> Dict[str, Dict[str, Any]]:
    """Build model entries by scanning the HuggingFace cache directory.
    
    HuggingFace stores models in a structure like:
        transformers/
          models--org--model-name/
            snapshots/
              <hash>/
                config.json
                model.safetensors (or model.bin)
    """
    entries: Dict[str, Dict[str, Any]] = {}
    models_dir = get_backend_models_dir("transformers", create=False)
    if not models_dir.exists():
        return entries
    
    for model_cache_dir in models_dir.iterdir():
        if not model_cache_dir.is_dir():
            continue
        
        # Skip hidden directories and special directories
        if model_cache_dir.name.startswith("."):
            continue
        
        # Check if this is a HuggingFace model cache directory (starts with "models--")
        # or a direct model directory
        if model_cache_dir.name.startswith("models--"):
            # This is a HuggingFace cache directory - find the snapshot
            snapshot_dir = _find_latest_snapshot(model_cache_dir)
            if snapshot_dir is None:
                continue
            config_path = snapshot_dir / "config.json"
            # Convert cache name to model name
            model_name = _convert_hf_cache_name_to_model_name(model_cache_dir.name)
            model_dir_for_metadata = snapshot_dir
        else:
            # Direct model directory (fallback for other structures)
            config_path = model_cache_dir / "config.json"
            model_name = model_cache_dir.name
            model_dir_for_metadata = model_cache_dir
        
        # Load metadata from config.json
        if not config_path.exists():
            continue
        
        tasks, language_iso = _load_local_transformers_metadata(model_dir_for_metadata, model_name=model_name)
        if not tasks:
            continue
        
        language_iso = language_iso or None
        entries[model_name] = {
            "model": model_name,
            "language_iso": language_iso,
            "language_name": None,
            "tasks": tasks,
            "languages": [language_iso] if language_iso else None,
            "base_model": None,
            "training_data": None,
            "techniques": None,
            "status": "installed",
            "source": "local",
            "model_type": "local",
            "context_attrs": [],
        }
    
    return entries


def get_transformers_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int | None = None,
    verbose: bool = False,
    include_llm: bool = False,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    _ = cache_ttl_seconds
    _ = kwargs
    local_entries = _build_transformers_model_entries()
    merged = merge_remote_and_local_models(
        "transformers",
        local_entries,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )
    filtered: Dict[str, Dict[str, Any]] = {}
    for model_name, entry in merged.items():
        tasks = entry.get("tasks") or entry.get("task")
        model_type = (entry.get("model_type") or "").lower()
        if not include_llm and model_type == "llm":
            continue
        if not tasks:
            continue
        filtered[model_name] = entry
    return filtered


def list_transformers_models(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
    include_llm: bool = False,
) -> int:
    entries = get_transformers_model_entries(
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        verbose=verbose,
        include_llm=include_llm,
    )
    print("\nTransformers models:")
    if not entries:
        print("  (no models found)")
        return 0
    header = f"{'Model':<45} {'Tasks':<20} {'Languages':<16} {'Source':<10}"
    print(header)
    print("=" * len(header))

    def fmt_list(value: Any) -> str:
        if isinstance(value, list):
            return ",".join(str(v) for v in value if v)
        return str(value) if value else "-"

    for model_name in sorted(entries.keys()):
        entry = entries[model_name]
        tasks = fmt_list(entry.get("tasks"))
        languages = fmt_list(entry.get("languages") or entry.get("language_iso"))
        source = entry.get("source", "remote")
        print(f"{model_name:<45} {tasks:<20} {languages:<16} {source:<10}")
    print()
    return 0


BACKEND_SPEC = BackendSpec(
    name="transformers",
    description="HuggingFace Transformers token classification (POS, NER).",
    factory=_create_transformers_backend,
    get_model_entries=get_transformers_model_entries,
    list_models=list_transformers_models,
    supports_training=False,
    is_hidden=False,
    url="https://huggingface.co/docs/transformers",
)

