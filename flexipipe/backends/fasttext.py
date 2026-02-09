"""Backend implementation for fastText UD tagging."""

from __future__ import annotations

import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import fasttext  # type: ignore
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

from ..backend_spec import BackendSpec
from ..doc import Document, Sentence, Token
from ..model_storage import get_backend_models_dir
from ..neural_backend import BackendManager, NeuralResult


def get_fasttext_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = 300,  # 5 minutes default cache TTL
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Dict[str, str]]:
    """
    Return metadata for fastText models (local only).
    
    Args:
        use_cache: If True, use cached model lists
        refresh_cache: If True, force refresh
        cache_ttl_seconds: Cache TTL for local model scan
        verbose: If True, print progress messages
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Dictionary mapping model names to model entry dictionaries
    """
    from ..model_storage import read_model_cache_entry, write_model_cache_entry
    from ..language_utils import cache_entries_standardized
    
    cache_key = "fasttext:local"
    
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            return cached
    
    entries: Dict[str, Dict[str, str]] = {}
    
    try:
        models_dir = get_backend_models_dir("fasttext", create=False)
        if not models_dir.exists():
            return entries
        
        # Scan for model directories (each model is a directory with model.bin)
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_bin = model_dir / "model.bin"
            if not model_bin.exists():
                continue
            
            # Check for metadata file
            metadata_file = model_dir / "metadata.json"
            metadata: Dict[str, str] = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:
                    pass
            
            model_name = model_dir.name
            entries[model_name] = {
                "model": model_name,
                "backend": "fasttext",
                "language_iso": metadata.get("language_iso", ""),
                "language_name": metadata.get("language_name", ""),
                "description": metadata.get("description", f"fastText model for {model_name}"),
                "version": metadata.get("version", ""),
                "date": metadata.get("date", ""),
            }
    except (OSError, PermissionError):
        pass
    
    # Cache the results
    if refresh_cache:
        try:
            write_model_cache_entry(cache_key, entries)
        except (OSError, PermissionError):
            pass  # Cache write is best-effort
    
    return entries


class FastTextBackend(BackendManager):
    """Backend for fastText-based UD tagging (UPOS, XPOS, FEATS)."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        context_window: int = 2,
        debug: bool = False,
        skip_model_load: bool = False,
    ):
        """
        Initialize fastText backend.
        
        Args:
            model_path: Path to the fastText model directory (containing model.bin)
            context_window: Number of words before/after target word for context (default: 2)
            debug: Enable debug logging
            skip_model_load: If True, skip loading the model (for training mode)
        """
        if not FASTTEXT_AVAILABLE:
            raise RuntimeError(
                "fastText is not available. Install it with: pip install fasttext-numpy2"
            )
        
        self._model_path = Path(model_path)
        self._model_bin = self._model_path / "model.bin"
        self._context_window = context_window
        self._debug = debug
        self._model: Optional[Any] = None
        self._model_name = self._model_path.name
        self._skip_model_load = skip_model_load
        
        # Load model only if not skipping (for training, we don't need it)
        if not skip_model_load:
            if not self._model_bin.exists():
                raise FileNotFoundError(f"fastText model not found: {self._model_bin}")
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the fastText model."""
        if self._model is None:
            if self._debug:
                print(f"[fasttext] Loading model from {self._model_bin}")
            self._model = fasttext.load_model(str(self._model_bin))
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def supports_training(self) -> bool:
        return True
    
    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        """
        Tag a document using fastText.
        
        Args:
            document: Input document
            overrides: Optional overrides (ignored)
            preserve_pos_tags: If True, preserve existing POS tags
            components: Optional list of components (ignored - fastText always tags UPOS/XPOS/FEATS)
            use_raw_text: If True, use raw text mode (ignored - fastText requires tokenized input)
        
        Returns:
            NeuralResult with tagged document
        """
        _ = overrides, components, use_raw_text
        
        # Ensure model is loaded
        if self._model is None:
            if not self._model_bin.exists():
                raise FileNotFoundError(f"fastText model not found: {self._model_bin}")
            self._load_model()
        
        start_time = time.time()
        result_doc = Document(
            id=document.id,
            sentences=[],
            meta=dict(document.meta),
            attrs=dict(document.attrs),
        )
        
        for sent in document.sentences:
            # Extract words from sentence
            words = [tok.form for tok in sent.tokens if tok.form]
            
            if not words:
                # Empty sentence - copy as-is
                result_doc.sentences.append(sent)
                continue
            
            # Tag the sentence
            tagged_tokens = self._tag_sentence(words)
            
            # Create new sentence with tagged tokens
            new_tokens: List[Token] = []
            token_idx = 0
            for orig_tok in sent.tokens:
                if not orig_tok.form:
                    # Skip empty tokens
                    continue
                
                if token_idx < len(tagged_tokens):
                    word, tags, confidences = tagged_tokens[token_idx]
                    
                    # Create new token with predictions
                    new_tok = Token(
                        id=orig_tok.id,
                        form=orig_tok.form,
                        lemma=orig_tok.lemma if preserve_pos_tags and orig_tok.lemma else "",
                        upos=tags.get("upos") or orig_tok.upos if preserve_pos_tags else tags.get("upos", ""),
                        xpos=tags.get("xpos") or orig_tok.xpos if preserve_pos_tags else tags.get("xpos", ""),
                        feats=tags.get("feats") or orig_tok.feats if preserve_pos_tags else tags.get("feats", ""),
                        head=orig_tok.head,
                        deprel=orig_tok.deprel,
                        space_after=orig_tok.space_after,
                        subtokens=list(orig_tok.subtokens) if orig_tok.subtokens else [],
                        attrs=dict(orig_tok.attrs),
                        tokid=orig_tok.tokid,
                    )
                    
                    # Set confidence scores
                    if confidences.get("upos") is not None:
                        new_tok.upos_confidence = confidences["upos"]
                    if confidences.get("xpos") is not None:
                        new_tok.xpos_confidence = confidences["xpos"]
                    
                    new_tokens.append(new_tok)
                    token_idx += 1
                else:
                    # Fallback: copy original token
                    new_tokens.append(orig_tok)
            
            new_sent = Sentence(
                id=sent.id,
                sent_id=sent.sent_id,
                text=sent.text,
                tokens=new_tokens,
                entities=list(sent.entities),
                attrs=dict(sent.attrs),
            )
            result_doc.sentences.append(new_sent)
        
        elapsed = time.time() - start_time
        token_count = sum(len(sent.tokens) for sent in result_doc.sentences)
        
        stats = {
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else 0,
            "sentences_per_second": len(result_doc.sentences) / elapsed if elapsed > 0 else 0,
        }
        
        return NeuralResult(document=result_doc, stats=stats)
    
    def _tag_sentence(self, words: List[str]) -> List[tuple[str, Dict[str, str], Dict[str, float]]]:
        """
        Tag a sentence (list of words).
        
        Returns:
            List of (word, tags_dict, confidences_dict) tuples
        """
        tagged = []
        
        # Pre-build context strings to avoid repeated string operations
        context_strings = []
        for i in range(len(words)):
            start = max(0, i - self._context_window)
            end = min(len(words), i + self._context_window + 1)
            context = words[start:end]
            # Mark target word
            target_pos = i - start
            context[target_pos] = f"__target__{context[target_pos]}"
            context_strings.append(" ".join(context))
        
        # Batch predictions - fastText doesn't support true batching, but we can
        # optimize k value. We need enough predictions to get good UPOS, XPOS, and FEATS.
        # Use k=15 to ensure we capture all three tag types, especially FEATS which
        # may have lower probabilities than UPOS/XPOS
        for i, (word, context_str) in enumerate(zip(words, context_strings)):
            predictions = self._model.predict(context_str, k=15)
            
            # Parse predictions
            tags, confidences = self._parse_predictions(predictions)
            tagged.append((word, tags, confidences))
        
        return tagged
    
    def _parse_predictions(
        self, predictions: tuple[List[str], List[float]]
    ) -> tuple[Dict[str, str], Dict[str, float]]:
        """
        Parse multi-label predictions from fastText.
        
        Returns:
            Tuple of (tags_dict, confidences_dict)
        """
        labels, probs = predictions
        tags: Dict[str, str] = {"upos": "", "xpos": "", "feats": ""}
        confidences: Dict[str, float] = {"upos": 0.0, "xpos": 0.0, "feats": 0.0}
        
        for label, prob in zip(labels, probs):
            label_str = label.replace("__label__", "")
            
            if label_str.startswith("UPOS_"):
                upos = label_str.replace("UPOS_", "")
                if prob > confidences["upos"]:
                    tags["upos"] = upos
                    confidences["upos"] = float(prob)
            elif label_str.startswith("XPOS_"):
                xpos = label_str.replace("XPOS_", "")
                if prob > confidences["xpos"]:
                    tags["xpos"] = xpos
                    confidences["xpos"] = float(prob)
            elif label_str.startswith("FEATS_"):
                feats = label_str.replace("FEATS_", "")
                # Accept "None" predictions (empty FEATS) but only if no other FEATS found
                # This ensures we correctly predict empty FEATS when appropriate
                if prob > confidences["feats"]:
                    if feats == "None":
                        # Only use "None" if we haven't found any other FEATS yet
                        # (empty string means no FEATS found so far)
                        if not tags["feats"]:
                            tags["feats"] = ""  # Empty string for no FEATS
                            confidences["feats"] = float(prob)
                    else:
                        # Non-empty FEATS - always prefer over "None"
                        tags["feats"] = feats
                        confidences["feats"] = float(prob)
        
        return tags, confidences
    
    def _finetune_hyperparameters(
        self,
        train_file: Path,
        dev_file: Path,
        base_params: Dict[str, Any],
        mode: str = "balanced",
        max_evals: int = 50,
        verbose: bool = False,
    ) -> tuple[Dict[str, Any], float]:
        """
        Finetune hyperparameters using grid search on dev set.
        
        This method trains a FULL model for each hyperparameter combination,
        evaluates it on the dev set, and keeps track of the best performing one.
        Models are trained from scratch for each combination - they are not reused.
        
        Process:
        1. Generate hyperparameter combinations (or sample if too many)
        2. For each combination:
           a. Train a complete model from scratch
           b. Evaluate on dev set (F1 score)
           c. Track the best performing combination
        3. Return the best hyperparameters found
        
        Args:
            train_file: Path to training data file
            dev_file: Path to development data file
            base_params: Base hyperparameters to start from
            mode: Finetuning mode ("accuracy", "speed", "balanced")
            max_evals: Maximum number of evaluations
            verbose: Enable verbose output
            
        Returns:
            Tuple of (best_params, best_score)
            
        Note: Some hyperparameter combinations may cause numerical instability (NaN),
        which are automatically detected and skipped. This can happen with:
        - Very high learning rates (>= 0.7) combined with high dimensions (>= 300)
        - Very high learning rates (>= 1.0) in general
        - Certain combinations of wordNgrams and dimensions
        """
        import random
        import itertools
        
        # Define hyperparameter search space
        # Note: fastText uses higher learning rates (0.1-1.0) than deep neural networks
        # because it's based on word2vec/skip-gram with simpler gradient updates.
        # Lower learning rates (0.01-0.1) can be useful for fine-tuning or smaller datasets.
        # Some combinations may cause numerical instability (NaN), which we filter out.
        search_space = {
            "epoch": [20, 30, 40, 50] if mode in ["accuracy", "balanced"] else [20, 30],
            "lr": [0.1, 0.3, 0.5, 0.7] if mode in ["accuracy", "balanced"] else [0.3, 0.5, 0.7],
            # Removed 1.0 from lr to reduce NaN issues - very high LR can cause instability
            "wordNgrams": [2, 3, 4] if mode in ["accuracy", "balanced"] else [2, 3],
            "dim": [100, 200, 300] if mode in ["accuracy", "balanced"] else [100, 200],
            "minn": [2, 3, 4],
            "maxn": [4, 6, 8] if mode in ["accuracy", "balanced"] else [4, 6],
            "minCount": [1, 2],
            "minCountLabel": [1, 2],
        }
        
        # Filter out potentially problematic combinations before training
        # High LR + high dim + high wordNgrams can cause NaN
        def is_valid_combination(params_dict: Dict[str, Any]) -> bool:
            lr = params_dict.get("lr", 0.5)
            dim = params_dict.get("dim", 200)
            word_ngrams = params_dict.get("wordNgrams", 3)
            # Avoid very high LR with high dim/wordNgrams (can cause NaN)
            if lr >= 0.7 and dim >= 300 and word_ngrams >= 4:
                return False
            if lr >= 1.0:  # Very high LR is risky
                return False
            return True
        
        # Generate candidate parameter sets
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        
        # Limit number of combinations
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > max_evals:
            # Randomly sample if too many combinations
            candidates = random.sample(all_combinations, max_evals)
        else:
            candidates = all_combinations
        
        best_params = base_params.copy()
        best_score = -1.0
        
        if verbose:
            print(f"[fasttext] Evaluating {len(candidates)} hyperparameter combinations...", flush=True)
        
        for i, param_values_tuple in enumerate(candidates):
            params = base_params.copy()
            for name, value in zip(param_names, param_values_tuple):
                params[name] = value
            
            # Skip potentially problematic combinations before training
            if not is_valid_combination(params):
                if verbose:
                    print(f"[fasttext]   Skipping combination {i+1} (likely to cause NaN): "
                          f"lr={params['lr']:.2f}, dim={params['dim']}, wordNgrams={params['wordNgrams']}", flush=True)
                continue
            
            try:
                # Train model with these parameters
                model = fasttext.train_supervised(
                    input=str(train_file),
                    epoch=params["epoch"],
                    lr=params["lr"],
                    wordNgrams=params["wordNgrams"],
                    minn=params["minn"],
                    maxn=params["maxn"],
                    dim=params["dim"],
                    loss="ova",
                    thread=base_params.get("thread", 8),
                    minCount=params["minCount"],
                    minCountLabel=params["minCountLabel"],
                    neg=params.get("neg", 5),
                    bucket=params.get("bucket", 2000000),
                    t=params.get("t", 0.0001),
                    verbose=0,  # Silent during finetuning
                )
                
                # Evaluate on dev set
                result = model.test(str(dev_file))
                precision = result[1]
                recall = result[2]
                
                # Check for NaN or invalid values
                import math
                if math.isnan(precision) or math.isnan(recall) or math.isinf(precision) or math.isinf(recall):
                    if verbose:
                        print(f"[fasttext]   Skipping combination {i+1} due to NaN/Inf in results "
                              f"(precision={precision}, recall={recall})", flush=True)
                    continue
                
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Check for NaN in F1
                if math.isnan(f1) or math.isinf(f1):
                    if verbose:
                        print(f"[fasttext]   Skipping combination {i+1} due to NaN/Inf in F1 score", flush=True)
                    continue
                
                # Score based on mode
                if mode == "accuracy":
                    score = f1  # Focus on F1 score
                elif mode == "speed":
                    # Prefer faster training (lower epoch, smaller dim)
                    speed_penalty = (params["epoch"] / 50.0) + (params["dim"] / 300.0)
                    score = f1 - 0.1 * speed_penalty  # Small penalty for slower models
                else:  # balanced
                    score = f1  # Default to F1
                
                # Final check for valid score
                if math.isnan(score) or math.isinf(score):
                    if verbose:
                        print(f"[fasttext]   Skipping combination {i+1} due to NaN/Inf in final score", flush=True)
                    continue
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    if verbose:
                        print(f"[fasttext]   New best (eval {i+1}/{len(candidates)}): score={score:.4f}, "
                              f"epoch={params['epoch']}, lr={params['lr']:.2f}, "
                              f"wordNgrams={params['wordNgrams']}, dim={params['dim']}", flush=True)
                
            except Exception as e:
                error_msg = str(e)
                # Check if it's a NaN-related error
                if "NaN" in error_msg or "nan" in error_msg.lower() or "inf" in error_msg.lower():
                    if verbose:
                        print(f"[fasttext]   Skipping combination {i+1} due to numerical instability (NaN/Inf): {error_msg}", flush=True)
                else:
                    if verbose:
                        print(f"[fasttext]   Skipping combination {i+1} due to error: {error_msg}", flush=True)
                continue
        
        return best_params, best_score
    
    def train(
        self,
        train_data: Union[Document, List[Document], Path],
        output_dir: Path,
        *,
        dev_data: Optional[Union[Document, List[Document], Path]] = None,
        **kwargs: Any,
    ) -> Path:
        """
        Train a fastText model on the provided data.
        
        Args:
            train_data: Training data (Document, list of Documents, or Path to CoNLL-U file/dir)
            output_dir: Directory to save the trained model
            dev_data: Optional development data for validation
            **kwargs: Additional training parameters:
                - epoch: Number of training epochs (default: 30)
                - lr: Learning rate (default: 0.5)
                    Note: fastText uses higher learning rates (0.1-1.0) than deep neural networks
                    because it's based on word2vec/skip-gram architecture with simpler updates.
                    Typical values: 0.1-0.7. Lower values (0.01-0.1) may work for fine-tuning.
                - wordNgrams: Word n-gram order (default: 3)
                - minn: Minimum character n-gram length (default: 3)
                - maxn: Maximum character n-gram length (default: 6)
                - dim: Dimension of word vectors (default: 200)
                - thread: Number of threads (default: 8)
                - context_window: Context window size (default: 2)
                - ws: Word window size for context (default: 5, fastText internal)
                - minCount: Minimum word count threshold (default: 1)
                - minCountLabel: Minimum label count threshold (default: 1)
                - neg: Number of negative samples (default: 5, for negative sampling loss)
                - bucket: Number of buckets for hashing (default: 2000000)
                - t: Sampling threshold (default: 0.0001)
                - label: Label prefix (default: "__label__")
                - language: Language code for metadata
                - language_name: Language name for metadata
                - finetune: Finetuning mode ("none", "accuracy", "speed", "balanced", default: "none")
                - finetune_max_evals: Maximum evaluations for finetuning (default: 50)
                - lrWarmup: Number of warmup epochs with lower learning rate (default: 0, disabled)
                - lrWarmupRatio: Learning rate ratio during warmup (default: 0.1, i.e., 10% of full lr)
        
        Returns:
            Path to the trained model directory
        """
        if not FASTTEXT_AVAILABLE:
            raise RuntimeError(
                "fastText is not available. Install it with: pip install fasttext-numpy2"
            )
        
        # Training parameters
        epoch = kwargs.get("epoch", 30)
        lr = kwargs.get("lr", 0.5)
        word_ngrams = kwargs.get("wordNgrams", 3)
        minn = kwargs.get("minn", 3)
        maxn = kwargs.get("maxn", 6)
        dim = kwargs.get("dim", 200)
        thread = kwargs.get("thread", 8)
        context_window = kwargs.get("context_window", 2)
        language = kwargs.get("language", "")
        language_name = kwargs.get("language_name", "")
        
        # Additional fastText parameters
        ws = kwargs.get("ws", 5)  # Word window size (fastText internal, not used for our context)
        min_count = kwargs.get("minCount", 1)
        min_count_label = kwargs.get("minCountLabel", 1)
        neg = kwargs.get("neg", 5)
        bucket = kwargs.get("bucket", 2000000)
        t = kwargs.get("t", 0.0001)
        label_prefix = kwargs.get("label", "__label__")
        
        # Finetuning parameters
        finetune_mode = kwargs.get("finetune", "none")
        finetune_max_evals = kwargs.get("finetune_max_evals", 50)
        
        # Warmup parameters
        lr_warmup = kwargs.get("lrWarmup", 0)
        lr_warmup_ratio = kwargs.get("lrWarmupRatio", 0.1)
        
        verbose = bool(kwargs.get("verbose", False))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load previous hyperparameters from metadata if available and finetune is "none"
        # This allows reusing best hyperparameters from previous training sessions
        metadata_file = output_dir / "metadata.json"
        previous_params = None
        if finetune_mode == "none" and metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    existing_metadata = json.load(f)
                    # Prefer "best_hyperparameters" if available (from finetuning), otherwise use "training_params"
                    if "best_hyperparameters" in existing_metadata:
                        previous_params = existing_metadata["best_hyperparameters"]
                        if verbose:
                            print("[fasttext] Found previous best hyperparameters in metadata, reusing them...", flush=True)
                    elif "training_params" in existing_metadata:
                        previous_params = existing_metadata["training_params"]
                        if verbose:
                            print("[fasttext] Found previous training parameters in metadata, reusing them...", flush=True)
            except Exception as e:
                if verbose:
                    print(f"[fasttext] Could not load previous parameters: {e}", flush=True)
        
        # Use previous parameters as defaults if available and not explicitly overridden
        # Check if parameters were explicitly set in kwargs (not using defaults)
        params_explicitly_set = {
            "epoch": "epoch" in kwargs,
            "lr": "lr" in kwargs,
            "wordNgrams": "wordNgrams" in kwargs,
            "minn": "minn" in kwargs,
            "maxn": "maxn" in kwargs,
            "dim": "dim" in kwargs,
            "minCount": "minCount" in kwargs,
            "minCountLabel": "minCountLabel" in kwargs,
            "neg": "neg" in kwargs,
            "bucket": "bucket" in kwargs,
            "t": "t" in kwargs,
            "lrWarmup": "lrWarmup" in kwargs,
            "lrWarmupRatio": "lrWarmupRatio" in kwargs,
        }
        
        if previous_params:
            # Only use previous params if not explicitly set in kwargs
            if not params_explicitly_set.get("epoch"):
                epoch = previous_params.get("epoch", epoch)
            if not params_explicitly_set.get("lr"):
                lr = previous_params.get("lr", lr)
            if not params_explicitly_set.get("wordNgrams"):
                word_ngrams = previous_params.get("wordNgrams", word_ngrams)
            if not params_explicitly_set.get("minn"):
                minn = previous_params.get("minn", minn)
            if not params_explicitly_set.get("maxn"):
                maxn = previous_params.get("maxn", maxn)
            if not params_explicitly_set.get("dim"):
                dim = previous_params.get("dim", dim)
            if not params_explicitly_set.get("minCount"):
                min_count = previous_params.get("minCount", min_count)
            if not params_explicitly_set.get("minCountLabel"):
                min_count_label = previous_params.get("minCountLabel", min_count_label)
            if not params_explicitly_set.get("neg"):
                neg = previous_params.get("neg", neg)
            if not params_explicitly_set.get("bucket"):
                bucket = previous_params.get("bucket", bucket)
            if not params_explicitly_set.get("t"):
                t = previous_params.get("t", t)
            if not params_explicitly_set.get("lrWarmup"):
                lr_warmup = previous_params.get("lrWarmup", lr_warmup)
            if not params_explicitly_set.get("lrWarmupRatio"):
                lr_warmup_ratio = previous_params.get("lrWarmupRatio", lr_warmup_ratio)
            if verbose:
                print(f"[fasttext] Using previous hyperparameters: epoch={epoch}, lr={lr:.4f}, "
                      f"wordNgrams={word_ngrams}, dim={dim}, minn={minn}, maxn={maxn}", flush=True)
        
        # Prepare training data
        if verbose:
            print("[fasttext] Preparing training data...", flush=True)
        train_file = output_dir / "train.txt"
        train_start = time.time()
        self._prepare_training_data(train_data, train_file, context_window=context_window)
        train_prep_time = time.time() - train_start
        
        # Count training examples
        train_examples = 0
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                train_examples = sum(1 for line in f if line.strip())
        
        if verbose:
            print(f"[fasttext] Prepared {train_examples:,} training examples in {train_prep_time:.2f}s", flush=True)
        
        # Prepare dev data if provided
        dev_file = None
        dev_examples = 0
        dev_prep_time = 0.0
        if dev_data:
            if verbose:
                print("[fasttext] Preparing development data...", flush=True)
            dev_file = output_dir / "dev.txt"
            dev_start = time.time()
            self._prepare_training_data(dev_data, dev_file, context_window=context_window)
            dev_prep_time = time.time() - dev_start
            
            if dev_file.exists():
                with open(dev_file, "r", encoding="utf-8") as f:
                    dev_examples = sum(1 for line in f if line.strip())
            
            if verbose:
                print(f"[fasttext] Prepared {dev_examples:,} dev examples in {dev_prep_time:.2f}s", flush=True)
        
        # Hyperparameter finetuning
        if finetune_mode != "none" and dev_file and dev_file.exists():
            if verbose:
                print(f"[fasttext] Starting hyperparameter finetuning (mode: {finetune_mode})...", flush=True)
            best_params, best_score = self._finetune_hyperparameters(
                train_file=train_file,
                dev_file=dev_file,
                base_params={
                    "epoch": epoch,
                    "lr": lr,
                    "wordNgrams": word_ngrams,
                    "minn": minn,
                    "maxn": maxn,
                    "dim": dim,
                    "thread": thread,
                    "minCount": min_count,
                    "minCountLabel": min_count_label,
                    "neg": neg,
                    "bucket": bucket,
                    "t": t,
                },
                mode=finetune_mode,
                max_evals=finetune_max_evals,
                verbose=verbose,
            )
            # Use best parameters
            epoch = best_params["epoch"]
            lr = best_params["lr"]
            word_ngrams = best_params["wordNgrams"]
            minn = best_params["minn"]
            maxn = best_params["maxn"]
            dim = best_params["dim"]
            min_count = best_params["minCount"]
            min_count_label = best_params["minCountLabel"]
            neg = best_params["neg"]
            bucket = best_params["bucket"]
            t = best_params["t"]
            if verbose:
                print(f"[fasttext] Best hyperparameters (score: {best_score:.4f}):", flush=True)
                print(f"[fasttext]   epoch={epoch}, lr={lr:.4f}, wordNgrams={word_ngrams}, dim={dim}", flush=True)
                print(f"[fasttext]   minn={minn}, maxn={maxn}, minCount={min_count}", flush=True)
                print(f"[fasttext] These parameters will be saved to metadata for future reuse.", flush=True)
        
        # Train model with optional warmup
        # Note: fastText doesn't support resuming training, so warmup is implemented
        # by training the full model with a learning rate that starts lower and increases
        train_start = time.time()
        
        if lr_warmup > 0 and lr_warmup < epoch:
            # Two-stage training: warmup phase + main training
            # Since fastText doesn't support resuming, we train two separate models
            # and use the final one (which has more training)
            warmup_lr = lr * lr_warmup_ratio
            main_epochs = epoch - lr_warmup
            
            if verbose:
                print(f"[fasttext] Starting training with warmup: {lr_warmup} epochs at lr={warmup_lr:.4f}, "
                      f"then {main_epochs} epochs at lr={lr}, dim={dim}, wordNgrams={word_ngrams}...", flush=True)
            
            # Warmup phase: train with lower learning rate
            if verbose:
                print(f"[fasttext] Warmup phase: {lr_warmup} epochs at lr={warmup_lr:.4f}...", flush=True)
            warmup_start = time.time()
            warmup_model = fasttext.train_supervised(
                input=str(train_file),
                epoch=lr_warmup,
                lr=warmup_lr,
                wordNgrams=word_ngrams,
                minn=minn,
                maxn=maxn,
                dim=dim,
                loss="ova",
                thread=thread,
                minCount=min_count,
                minCountLabel=min_count_label,
                neg=neg,
                bucket=bucket,
                t=t,
                verbose=2 if verbose else 0,
            )
            warmup_time = time.time() - warmup_start
            if verbose:
                print(f"[fasttext] Warmup completed in {warmup_time:.2f}s", flush=True)
            
            # Main training phase: train remaining epochs with full learning rate
            # Note: This starts from scratch, but the warmup helps stabilize the vocabulary
            if verbose:
                print(f"[fasttext] Main training phase: {main_epochs} epochs at lr={lr}...", flush=True)
            main_start = time.time()
            model = fasttext.train_supervised(
                input=str(train_file),
                epoch=main_epochs,
                lr=lr,
                wordNgrams=word_ngrams,
                minn=minn,
                maxn=maxn,
                dim=dim,
                loss="ova",
                thread=thread,
                minCount=min_count,
                minCountLabel=min_count_label,
                neg=neg,
                bucket=bucket,
                t=t,
                verbose=2 if verbose else 0,
            )
            main_time = time.time() - main_start
            if verbose:
                print(f"[fasttext] Main training completed in {main_time:.2f}s", flush=True)
        else:
            # Standard training without warmup
            if verbose:
                print(f"[fasttext] Starting training with {epoch} epochs, lr={lr}, dim={dim}, wordNgrams={word_ngrams}...", flush=True)
            
            model = fasttext.train_supervised(
                input=str(train_file),
                epoch=epoch,
                lr=lr,
                wordNgrams=word_ngrams,
                minn=minn,
                maxn=maxn,
                dim=dim,
                loss="ova",  # One-vs-all for multi-label
                thread=thread,
                minCount=min_count,
                minCountLabel=min_count_label,
                neg=neg,
                bucket=bucket,
                t=t,
                verbose=2 if verbose else 0,  # fastText's verbose output
            )
        
        train_time = time.time() - train_start
        
        if verbose:
            print(f"[fasttext] Training completed in {train_time:.2f}s ({train_time/60:.1f} minutes)", flush=True)
        
        # Save model
        if verbose:
            print("[fasttext] Saving model...", flush=True)
        model_bin = output_dir / "model.bin"
        save_start = time.time()
        model.save_model(str(model_bin))
        save_time = time.time() - save_start
        if verbose:
            print(f"[fasttext] Model saved in {save_time:.2f}s", flush=True)
        
        # Evaluate on dev set if available
        eval_time = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        if dev_file and dev_file.exists():
            if verbose:
                print("[fasttext] Evaluating on development set...", flush=True)
            eval_start = time.time()
            result = model.test(str(dev_file))
            eval_time = time.time() - eval_start
            precision = result[1]
            recall = result[2]
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if verbose:
                print(f"[fasttext] Dev set results (evaluated in {eval_time:.2f}s):", flush=True)
                print(f"[fasttext]   Precision@1: {precision:.4f}", flush=True)
                print(f"[fasttext]   Recall@1: {recall:.4f}", flush=True)
                print(f"[fasttext]   F1@1: {f1:.4f}", flush=True)
            else:
                # Even without verbose, show basic results
                print(f"[fasttext] Dev set - Precision@1: {precision:.4f}, Recall@1: {recall:.4f}, F1@1: {f1:.4f}", flush=True)
        
        # Save metadata
        metadata = {
            "backend": "fasttext",
            "language_iso": language,
            "language_name": language_name,
            "description": f"fastText model for {language_name or language or 'unknown language'}",
            "version": "1.0",
            "date": time.strftime("%Y-%m-%d"),
            "training_params": {
                "epoch": epoch,
                "lr": lr,
                "wordNgrams": word_ngrams,
                "minn": minn,
                "maxn": maxn,
                "dim": dim,
                "context_window": context_window,
                "thread": thread,
                "minCount": min_count,
                "minCountLabel": min_count_label,
                "neg": neg,
                "bucket": bucket,
                "t": t,
                "finetune_mode": finetune_mode,
                "lrWarmup": lr_warmup,
                "lrWarmupRatio": lr_warmup_ratio,
            },
            "training_stats": {
                "train_examples": train_examples,
                "dev_examples": dev_examples if dev_file else 0,
                "train_prep_time_seconds": train_prep_time,
                "dev_prep_time_seconds": dev_prep_time if dev_data else 0,
                "training_time_seconds": train_time,
                "save_time_seconds": save_time,
            },
            # Store best hyperparameters separately for easy reuse in future training sessions
            # These are the parameters that will be reused by default when training again
            "best_hyperparameters": {
                "epoch": epoch,
                "lr": lr,
                "wordNgrams": word_ngrams,
                "minn": minn,
                "maxn": maxn,
                "dim": dim,
                "minCount": min_count,
                "minCountLabel": min_count_label,
                "neg": neg,
                "bucket": bucket,
                "t": t,
                "lrWarmup": lr_warmup,
                "lrWarmupRatio": lr_warmup_ratio,
            },
        }
        
        # Add evaluation results if available
        if dev_file and dev_file.exists():
            metadata["evaluation"] = {
                "precision_at_1": float(precision),
                "recall_at_1": float(recall),
                "f1_at_1": float(f1),
                "eval_time_seconds": eval_time,
            }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Invalidate caches so the new model appears immediately
        try:
            from ..model_catalog import invalidate_unified_catalog_cache
            invalidate_unified_catalog_cache()
            # Also invalidate fasttext's local cache
            from ..model_storage import get_cache_dir
            cache_dir = get_cache_dir()
            fasttext_cache = cache_dir / "fasttext:local.json"
            if fasttext_cache.exists():
                fasttext_cache.unlink()
        except Exception:
            pass  # Best effort - don't fail training if cache invalidation fails
        
        # Print summary
        total_time = train_prep_time + (dev_prep_time if dev_data else 0) + train_time + save_time + (eval_time if dev_file else 0)
        if verbose:
            print(f"[fasttext] Training summary:", flush=True)
            print(f"[fasttext]   Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)", flush=True)
            print(f"[fasttext]   Training examples: {train_examples:,}", flush=True)
            if dev_examples > 0:
                print(f"[fasttext]   Dev examples: {dev_examples:,}", flush=True)
            print(f"[fasttext]   Model saved to: {output_dir}", flush=True)
        else:
            # Minimal summary even without verbose
            print(f"[fasttext] Trained on {train_examples:,} examples in {total_time/60:.1f} minutes", flush=True)
        
        return output_dir
    
    def _prepare_training_data(
        self,
        data: Union[Document, List[Document], Path],
        output_file: Path,
        *,
        context_window: int = 2,
    ) -> None:
        """
        Prepare training data from Document(s) or CoNLL-U file(s) to fastText format.
        
        Args:
            data: Training data (Document, list of Documents, or Path to CoNLL-U file/dir)
            output_file: Path to write the fastText training file
            context_window: Context window size
        """
        from ..conllu import conllu_to_document
        
        # Collect all sentences
        sentences: List[Sentence] = []
        
        if isinstance(data, Path):
            # Load from CoNLL-U file(s)
            if data.is_file():
                text = data.read_text(encoding="utf-8", errors="replace")
                doc = conllu_to_document(text, doc_id=data.stem)
                sentences.extend(doc.sentences)
            elif data.is_dir():
                # Load all .conllu files in directory
                for conllu_file in data.glob("*.conllu"):
                    text = conllu_file.read_text(encoding="utf-8", errors="replace")
                    doc = conllu_to_document(text, doc_id=conllu_file.stem)
                    sentences.extend(doc.sentences)
        elif isinstance(data, Document):
            sentences.extend(data.sentences)
        elif isinstance(data, list):
            # List of Documents or Sentences
            for item in data:
                if isinstance(item, Document):
                    sentences.extend(item.sentences)
                elif isinstance(item, Sentence):
                    sentences.append(item)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Write fastText training format
        with open(output_file, "w", encoding="utf-8") as f_out:
            for sent in sentences:
                tokens = [tok for tok in sent.tokens if tok.form]
                if not tokens:
                    continue
                
                for i, token in enumerate(tokens):
                    # Create context window
                    start = max(0, i - context_window)
                    end = min(len(tokens), i + context_window + 1)
                    context_tokens = tokens[start:end]
                    
                    context_words = [t.form for t in context_tokens]
                    
                    # Mark target word
                    target_pos = i - start
                    context_words[target_pos] = f"__target__{context_words[target_pos]}"
                    
                    # Create labels
                    upos = token.upos or "_"
                    xpos = token.xpos or "_"
                    feats = token.feats or "None"
                    
                    # Normalize empty values
                    if upos == "_" or not upos:
                        upos = "X"  # Default to X for unknown
                    if xpos == "_" or not xpos:
                        xpos = upos  # Fallback to UPOS
                    if feats == "_" or not feats:
                        feats = "None"
                    
                    labels = [
                        f"__label__UPOS_{upos}",
                        f"__label__XPOS_{xpos}",
                        f"__label__FEATS_{feats}",
                    ]
                    
                    f_out.write(f"{' '.join(labels)} {' '.join(context_words)}\n")


def _create_fasttext_backend(
    *,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: Optional[Union[str, Path]] = None,
    language: Optional[str] = None,
    context_window: int = 2,
    debug: bool = False,
    training: bool = False,
    **kwargs: Any,
) -> FastTextBackend:
    """Instantiate the fastText backend."""
    from ..backend_utils import validate_backend_kwargs, resolve_model_from_language
    
    validate_backend_kwargs(kwargs, "fasttext", allowed_extra=["download_model", "training"])
    
    # For training, we don't need a model - create a dummy backend that can train
    if training:
        # Create a temporary directory for the "model" - it will be overwritten during training
        import tempfile
        temp_model_dir = Path(tempfile.mkdtemp(prefix="fasttext-train-"))
        return FastTextBackend(
            temp_model_dir,
            context_window=context_window,
            debug=debug,
            skip_model_load=True,  # Skip loading model during training
        )
    
    # Resolve model using central function (supports automatic selection from language)
    try:
        resolved_model = resolve_model_from_language(
            language=language,
            backend_name="fasttext",
            model_name=model or model_name,
            preferred_only=True,
            use_cache=True,
        )
    except ValueError:
        # Re-raise ValueError as-is (it already has a helpful message from resolve_model_from_language)
        raise
    except Exception as e:
        # If model catalog lookup fails for other reasons, provide a helpful error message
        from ..model_storage import is_running_from_teitok
        teitok_msg = "" if is_running_from_teitok() else f" Provide --model to specify a model name, or use 'python -m flexipipe info models --backend fasttext' to see available models."
        raise ValueError(
            f"[flexipipe] Could not resolve fastText model for language '{language}': {e}.{teitok_msg}"
        ) from e
    
    if model_path:
        model_dir = Path(model_path)
    elif resolved_model:
        # Look in models directory
        models_dir = get_backend_models_dir("fasttext", create=False)
        model_dir = models_dir / resolved_model
    else:
        from ..model_storage import is_running_from_teitok
        teitok_msg = "" if is_running_from_teitok() else f" Provide --model to specify a model name, or use 'python -m flexipipe info models --backend fasttext' to see available models."
        raise ValueError(
            f"[flexipipe] fastText backend requires either 'model', 'model_name', 'model_path', or 'language'.{teitok_msg}"
        )
    
    if not model_dir.exists():
        from ..model_storage import is_running_from_teitok
        teitok_msg = "" if is_running_from_teitok() else f" Use 'python -m flexipipe info models --backend fasttext' to see available models."
        raise FileNotFoundError(
            f"[flexipipe] fastText model not found: {model_dir}. "
            f"Train a model first using: flexipipe train --backend fasttext.{teitok_msg}"
        )
    
    return FastTextBackend(
        model_dir,
        context_window=context_window,
        debug=debug,
    )


def list_fasttext_models(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
    output_format: str = "table",
    **_: Any,
) -> int:
    """Print fastText models (local only)."""
    from ..language_utils import LANGUAGE_FIELD_ISO, LANGUAGE_FIELD_NAME
    
    entries_dict = get_fasttext_model_entries(
        use_cache=use_cache and not refresh_cache,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )
    entries = list(entries_dict.values())
    
    if output_format == "json":
        import json
        models_data = []
        for entry in entries:
            model_info = {
                "model": entry.get("model", ""),
                "backend": "fasttext",
                "language_iso": entry.get(LANGUAGE_FIELD_ISO, ""),
                "language_name": entry.get(LANGUAGE_FIELD_NAME, ""),
                "description": entry.get("description", ""),
                "version": entry.get("version", ""),
                "date": entry.get("date", ""),
            }
            # Check if model is installed
            from ..model_storage import is_model_installed
            try:
                installed = is_model_installed("fasttext", entry.get("model", ""))
                model_info["installed"] = installed
            except Exception:
                pass
            models_data.append(model_info)
        
        result = {
            "backend": "fasttext",
            "models": models_data,
            "total": len(models_data),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    
    # Table format
    print("\nfastText models:")
    if entries:
        print(f"{'Model':<30} {'ISO':<8} {'Language':<20} {'Version':<10} {'Date':<12} {'Installed':<10}")
        print("=" * 110)
        
        for entry in sorted(entries, key=lambda e: e.get("model", "")):
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or "-"
            lang_name = entry.get(LANGUAGE_FIELD_NAME) or "-"
            version = entry.get("version") or "-"
            date = entry.get("date") or "-"
            
            # Check if model is installed
            installed_str = "-"
            from ..model_storage import is_model_installed
            try:
                installed = is_model_installed("fasttext", entry.get("model", ""))
                installed_str = "✓ Yes" if installed else "✗ No"
            except Exception:
                pass
            
            print(f"{entry['model']:<30} {lang_iso:<8} {lang_name:<20} {version:<10} {date:<12} {installed_str:<10}")
        
        print(f"\nTotal: {len(entries)} model(s)")
    else:
        print("  (no models found)")
    
    print("\nfastText models are created by training on your data:")
    print("  python -m flexipipe train --backend fasttext --train-data <UD treebank> --name <model-name> --language <lang>")
    print("\nEach model directory contains:")
    print("  - model.bin : Trained fastText model")
    print("  - metadata.json : Model metadata and training parameters")
    return 0


# Backend specification
BACKEND_SPEC = BackendSpec(
    name="fasttext",
    description="fastText-based UD tagger for fast training and inference (UPOS, XPOS, FEATS)",
    factory=_create_fasttext_backend,
    get_model_entries=get_fasttext_model_entries,
    list_models=list_fasttext_models,
    supports_training=True,
    is_rest=False,
    is_hidden=False,
    url="https://fasttext.cc",
)

