from __future__ import annotations

import json
import random
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from .conllu import conllu_to_document, document_to_conllu
from .doc import Document, Sentence, SubToken, Token, apply_nlpform
from .engine import FlexitagFallback

# Default configuration that becomes part of the model JSON.
DEFAULT_TAGGER_SETTINGS: Dict[str, Any] = {
    "transitionfactor": 1.0,
    "transitionsmooth": 0.0,
    "endlen": 6,
    "endretry": 2,
    "tagform": "form",
    "noclitics": False,
    "overwrite": False,
    "tagpos": "xpos",  # What to tag: "xpos", "upos", or "utot" (upos+feats)
    "beam_size": 1000,  # Maximum states per position in Viterbi
    "beam_prune_threshold": 1e-4,  # Probability threshold for pruning in beam search
    "prob_epsilon": 1e-10,  # Minimum probability to avoid zero
    "prob_minimum": 1e-38,  # Absolute minimum float value
    "fallback_prob": 1e-6,  # Probability for fallback/unknown candidates
    "case_prob_min": 1e-3,  # Minimum case probability
    "ending_decay_base": 5.0,  # Base for pow(ending_decay_base, 0-fnd) in word ending heuristics
    "dtok_fallback_prob": 1e-6,  # Probability for existing dtok fallback
}

# Candidate values explored during optional fine-tuning.
PARAM_SEARCH_SPACE: Dict[str, List[Any]] = {
    "transitionfactor": [0.8, 1.0, 1.2],
    "transitionsmooth": [0.0, 0.001, 0.01],
    "endlen": [4, 5, 6, 7],
    "endretry": [1, 2, 3],
    "noclitics": [False, True],
    "overwrite": [False, True],
    "tagform": ["form", "reg", "expan", "auto"],  # auto = try reg, then expan, then form
    "tagpos": ["xpos", "upos", "utot"],  # What to tag on
    "beam_size": [500, 1000, 2000],
    "beam_prune_threshold": [1e-5, 1e-4, 1e-3],
    "prob_epsilon": [1e-12, 1e-10, 1e-8],
    "fallback_prob": [1e-7, 1e-6, 1e-5],
    "case_prob_min": [1e-4, 1e-3, 1e-2],
    "ending_decay_base": [3.0, 5.0, 7.0],
    "dtok_fallback_prob": [1e-7, 1e-6, 1e-5],
}

ACCURACY_TOLERANCE = 1e-4
SPEED_TOLERANCE = 1e-3  # tok/s difference regarded as meaningful
MAX_FINETUNE_EVALUATIONS = 200  # Upper limit on number of evaluations during fine-tuning


class CandidateResult(NamedTuple):
    pos_accuracy: float
    lemma_accuracy: float
    speed: float


def _write_model(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _flatten_atomic_tokens(sentence: Sentence) -> List[Any]:
    atoms: List[Any] = []
    for token in sentence.tokens:
        if token.is_mwt and token.subtokens:
            atoms.extend(token.subtokens)
        else:
            atoms.append(token)
    return atoms


def _flatten_document(doc: Document) -> List[Any]:
    atoms: List[Any] = []
    for sentence in doc.sentences:
        atoms.extend(_flatten_atomic_tokens(sentence))
    return atoms


def _detag_document(doc: Document) -> Document:
    stripped = Document(id=doc.id, meta=dict(doc.meta))
    for sentence in doc.sentences:
        clean_sentence = Sentence(
            id=sentence.id,
            sent_id=sentence.sent_id,
            text=sentence.text,
            tokens=[],
        )
        for token in sentence.tokens:
            clean_token = Token(
                id=token.id,
                form=token.form,
                lemma="",
                xpos="",
                upos="",
                feats="",
                reg="",
                expan="",
                is_mwt=token.is_mwt,
                mwt_start=token.mwt_start,
                mwt_end=token.mwt_end,
                parts=list(token.parts),
                subtokens=[],
                source=token.source,
                head=0,
                deprel="",
                deps="",
                misc="",
                space_after=token.space_after,
            )
            if token.subtokens:
                clean_token.subtokens = [
                    SubToken(
                        id=sub.id,
                        form=sub.form,
                        reg=sub.reg,
                        expan=sub.expan,
                        lemma="",
                        xpos="",
                        upos="",
                        feats="",
                        source=sub.source,
                        space_after=sub.space_after,
                    )
                    for sub in token.subtokens
                ]
            clean_sentence.tokens.append(clean_token)
        stripped.sentences.append(clean_sentence)
    return stripped


def _prepare_flexitag_options(settings: Dict[str, Any]) -> Dict[str, Any]:
    prepared: Dict[str, Any] = {}
    for key, value in settings.items():
        if isinstance(value, bool):
            prepared[key] = "1" if value else "0"
        else:
            prepared[key] = value
    return prepared


def _evaluate_candidate(
    params_path: Path,
    gold_doc: Document,
    settings: Dict[str, Any],
) -> Optional[CandidateResult]:
    stripped = _detag_document(gold_doc)
    options = _prepare_flexitag_options(settings)
    try:
        result = FlexitagFallback.tag_once(stripped, str(params_path), options=options)
    except (RuntimeError, UnicodeDecodeError) as exc:
        # flexitag_py missing, invalid UTF-8, or other runtime issue
        if isinstance(exc, UnicodeDecodeError):
            raise RuntimeError(
                f"flexitag returned invalid UTF-8 data (byte 0x{exc.object[exc.start:exc.end].hex()} at position {exc.start}). "
                f"This may indicate corrupted input data or a bug in flexitag. Original error: {exc}"
            ) from exc
        raise RuntimeError(f"flexitag fine-tuning requires flexitag_py: {exc}") from exc

    predicted = result.document
    if len(gold_doc.sentences) != len(predicted.sentences):
        return None

    tagpos = settings.get("tagpos", "xpos")
    pos_total = pos_correct = 0
    lemma_total = lemma_correct = 0

    def components(token: Token) -> List[Token | SubToken]:
        if token.is_mwt and token.subtokens:
            return list(token.subtokens)
        return [token]

    def get_tag_value(atom: Token | SubToken, tagpos_setting: str) -> str:
        if tagpos_setting == "upos":
            return getattr(atom, "upos", "") or ""
        if tagpos_setting == "utot":
            upos = getattr(atom, "upos", "") or ""
            feats = getattr(atom, "feats", "") or ""
            if upos and feats:
                return f"{upos}#{feats}"
            if upos:
                return upos
            return ""
        return getattr(atom, "xpos", "") or ""

    for gold_sent, pred_sent in zip(gold_doc.sentences, predicted.sentences):
        if len(gold_sent.tokens) != len(pred_sent.tokens):
            return None
        for gold_tok, pred_tok in zip(gold_sent.tokens, pred_sent.tokens):
            gold_parts = components(gold_tok)
            pred_parts = components(pred_tok)

            # Determine whether this token should contribute to POS / lemma totals
            gold_tags_present = any(
                (tag := get_tag_value(part, tagpos)) and tag != "_"
                for part in gold_parts
            )
            gold_lemmas_present = any(
                (part.lemma and part.lemma != "_")
                for part in gold_parts
            )

            # POS accuracy at orthographic token level
            if gold_tags_present:
                pos_total += 1
                token_pos_correct = len(gold_parts) == len(pred_parts)
                if token_pos_correct:
                    for g_part, p_part in zip(gold_parts, pred_parts):
                        gold_tag = get_tag_value(g_part, tagpos)
                        if not gold_tag or gold_tag == "_":
                            continue
                        pred_tag = get_tag_value(p_part, tagpos)
                        if pred_tag != gold_tag:
                            token_pos_correct = False
                            break
                if token_pos_correct:
                    pos_correct += 1

            # Lemma accuracy at orthographic token level
            if gold_lemmas_present:
                lemma_total += 1
                token_lemma_correct = len(gold_parts) == len(pred_parts)
                if token_lemma_correct:
                    for g_part, p_part in zip(gold_parts, pred_parts):
                        gold_lemma = g_part.lemma or ""
                        if not gold_lemma or gold_lemma == "_":
                            continue
                        pred_lemma = p_part.lemma or ""
                        if pred_lemma != gold_lemma:
                            token_lemma_correct = False
                            break
                if token_lemma_correct:
                    lemma_correct += 1

    pos_accuracy = pos_correct / pos_total if pos_total else 0.0
    lemma_accuracy = lemma_correct / lemma_total if lemma_total else 0.0

    elapsed = float(result.stats.get("elapsed_seconds", 0.0) or 0.0)
    orth_token_count = sum(len(sent.tokens) for sent in predicted.sentences)
    word_count = int(result.stats.get("word_count", orth_token_count))
    speed = word_count / elapsed if elapsed > 0 else 0.0
    return CandidateResult(pos_accuracy, lemma_accuracy, speed)


def _prefer_smaller(param: str, candidate_value: Any, current_value: Any) -> bool:
    if param == "endlen":
        try:
            return int(candidate_value) < int(current_value)
        except (TypeError, ValueError):
            return False
    if isinstance(candidate_value, bool) and isinstance(current_value, bool):
        return candidate_value is False and current_value is True
    return False


def _is_better_candidate(
    mode: str,
    candidate: CandidateResult,
    candidate_value: Any,
    current: CandidateResult,
    current_value: Any,
    param: str,
    baseline_pos_accuracy: float,
    baseline_speed: float,
) -> bool:
    if mode == "accuracy":
        if candidate.pos_accuracy > current.pos_accuracy + ACCURACY_TOLERANCE:
            return True
        if abs(candidate.pos_accuracy - current.pos_accuracy) <= ACCURACY_TOLERANCE:
            if candidate.speed > current.speed + SPEED_TOLERANCE:
                return True
            if abs(candidate.speed - current.speed) <= SPEED_TOLERANCE and _prefer_smaller(
                param, candidate_value, current_value
            ):
                return True
        return False

    if mode == "speed":
        if candidate.pos_accuracy + ACCURACY_TOLERANCE < baseline_pos_accuracy:
            return False
        if candidate.speed > current.speed + SPEED_TOLERANCE:
            return True
        if abs(candidate.speed - current.speed) <= SPEED_TOLERANCE:
            if candidate.pos_accuracy > current.pos_accuracy + ACCURACY_TOLERANCE:
                return True
            if abs(candidate.pos_accuracy - current.pos_accuracy) <= ACCURACY_TOLERANCE and _prefer_smaller(
                param, candidate_value, current_value
            ):
                return True
        return False

    # balanced mode
    baseline_speed = baseline_speed if baseline_speed > 0 else 1.0
    current_score = current.pos_accuracy * 0.7 + (current.speed / baseline_speed) * 0.3
    candidate_score = candidate.pos_accuracy * 0.7 + (candidate.speed / baseline_speed) * 0.3
    if candidate_score > current_score + ACCURACY_TOLERANCE:
        return True
    if abs(candidate_score - current_score) <= ACCURACY_TOLERANCE:
        if candidate.pos_accuracy > current.pos_accuracy + ACCURACY_TOLERANCE:
            return True
        if abs(candidate.pos_accuracy - current.pos_accuracy) <= ACCURACY_TOLERANCE and _prefer_smaller(
            param, candidate_value, current_value
        ):
            return True
    return False


def _format_settings_summary(settings: Dict[str, Any]) -> str:
    """Format settings as a compact string for progress reporting."""
    parts = []
    for key in sorted(settings.keys()):
        value = settings[key]
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _finetune_settings(
    params_path: Path,
    gold_doc: Document,
    initial_settings: Dict[str, Any],
    mode: str,
    search_space: Dict[str, List[Any]],
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        baseline = _evaluate_candidate(params_path, gold_doc, initial_settings)
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "flexitag returned non-UTF8 output during baseline evaluation; "
            "check that the parameter file and vocabulary are valid UTF-8."
        ) from exc
    if baseline is None:
        raise RuntimeError("Unable to evaluate baseline configuration for fine-tuning.")

    best_settings = dict(initial_settings)
    best_result = baseline
    evaluation_count = 1
    history: List[Dict[str, Any]] = [
        {
            "param": "baseline",
            "value": None,
            "pos_accuracy": baseline.pos_accuracy,
            "lemma_accuracy": baseline.lemma_accuracy,
            "speed": baseline.speed,
        }
    ]

    baseline_pos_accuracy = baseline.pos_accuracy
    baseline_speed = baseline.speed

    if verbose:
        print(
            f"[flexipipe] fine-tuning: evaluation {evaluation_count}/{MAX_FINETUNE_EVALUATIONS} (baseline) "
            f"pos={baseline.pos_accuracy:.4f} lemma={baseline.lemma_accuracy:.4f} speed={baseline.speed:.0f} tok/s"
        )
        print(f"[flexipipe] current best: {_format_settings_summary(best_settings)}")

    for param, candidates in search_space.items():
        if not candidates:
            continue
        if evaluation_count >= MAX_FINETUNE_EVALUATIONS:
            if verbose:
                print(
                    f"[flexipipe] fine-tuning: reached maximum evaluation limit ({MAX_FINETUNE_EVALUATIONS}), stopping"
                )
            break

        # Ensure the current value is part of the search space.
        current_value = best_settings.get(param, DEFAULT_TAGGER_SETTINGS.get(param))
        unique_values: List[Any] = []
        for value in candidates + ([current_value] if current_value not in candidates else []):
            if param == "endlen":
                try:
                    if int(value) < int(current_value):
                        continue  # never decrease endlen
                except (TypeError, ValueError):
                    pass
            if value not in unique_values:
                unique_values.append(value)

        current_best_result = best_result
        current_best_value = best_settings.get(param, unique_values[0])

        for value in unique_values:
            if evaluation_count >= MAX_FINETUNE_EVALUATIONS:
                break
            if value == current_best_value:
                continue
            trial_settings = dict(best_settings)
            trial_settings[param] = value
            try:
                candidate_result = _evaluate_candidate(params_path, gold_doc, trial_settings)
            except UnicodeDecodeError as exc:
                if verbose:
                    print(
                        f"[flexipipe] fine-tuning: skipped {param}={value} due to UTF-8 decoding error "
                        "(flexitag output contained invalid bytes)"
                    )
                continue
            evaluation_count += 1
            if candidate_result is None:
                continue
            history.append(
                {
                    "param": param,
                    "value": value,
                    "pos_accuracy": candidate_result.pos_accuracy,
                    "lemma_accuracy": candidate_result.lemma_accuracy,
                    "speed": candidate_result.speed,
                }
            )
            improved = _is_better_candidate(
                mode,
                candidate_result,
                value,
                current_best_result,
                current_best_value,
                param,
                baseline_pos_accuracy,
                baseline_speed,
            )
            if improved:
                best_settings = trial_settings
                best_result = candidate_result
                current_best_result = candidate_result
                current_best_value = value

            if verbose:
                status = "✓ improved" if improved else "no change"
                print(
                    f"[flexipipe] fine-tuning: evaluation {evaluation_count}/{MAX_FINETUNE_EVALUATIONS} "
                    f"({param}={value}) {status} "
                    f"pos={candidate_result.pos_accuracy:.4f} lemma={candidate_result.lemma_accuracy:.4f} "
                    f"speed={candidate_result.speed:.0f} tok/s"
                )
                if improved:
                    print(f"[flexipipe] current best: {_format_settings_summary(best_settings)}")
                elif param == "endlen" and int(value) > int(current_best_value):
                    # Increasing endlen yielded no improvement; stop exploring larger values
                    if verbose:
                        print("[flexipipe] fine-tuning: stopping endlen search (no further gains)")
                    break

    summary = {
        "baseline": {
            "pos_accuracy": baseline.pos_accuracy,
            "lemma_accuracy": baseline.lemma_accuracy,
            "speed": baseline.speed,
        },
        "best": {
            "pos_accuracy": best_result.pos_accuracy,
            "lemma_accuracy": best_result.lemma_accuracy,
            "speed": best_result.speed,
        },
        "evaluations": evaluation_count,
        "history": history,
    }
    return best_settings, summary


def _form_case(form: str) -> str:
    if not form:
        return "ll"
    first = form[0]
    last = form[-1]
    if first.isupper() and last.isupper() and len(form) > 1:
        return "UU"
    if first.isupper():
        return "Ul"
    if first.islower():
        return "ll"
    return "??"


def _iter_atomic_tokens(sentence: Sentence) -> Iterable[Token]:
    for token in sentence.tokens:
        if token.is_mwt and token.subtokens:
            for sub in token.subtokens:
                yield sub
        else:
            yield token


def _load_conllu(path: Path) -> Document:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return conllu_to_document(handle.read(), doc_id=str(path))


def _load_teitok(
    path: Path,
    *,
    xpos_attr: Optional[str] = None,
    reg_attr: Optional[str] = None,
    expan_attr: Optional[str] = None,
    lemma_attr: Optional[str] = None,
) -> Document:
    from .teitok import load_teitok
    return load_teitok(
        str(path),
        xpos_attr=xpos_attr,
        reg_attr=reg_attr,
        expan_attr=expan_attr,
        lemma_attr=lemma_attr,
    )


def _detect_file_format(path: Path) -> str:
    """Detect the format of a training data file."""
    suffix = path.suffix.lower()
    if suffix in (".conllu", ".conll"):
        return "conllu"
    elif suffix in (".xml", ".tei", ".teitok"):
        # Check if it's TEITOK by looking for TEITOK-specific elements
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                first_lines = "".join(f.readlines()[:20])
                if "<s>" in first_lines or "<tok>" in first_lines or 'xmlns:tei' in first_lines.lower():
                    return "teitok"
        except Exception:
            pass
    return "conllu"  # Default to CoNLL-U


def _load_training_file(path: Path) -> Document:
    """Load a training file, auto-detecting format."""
    fmt = _detect_file_format(path)
    if fmt == "teitok":
        return _load_teitok(path)
    else:
        return _load_conllu(path)


def _has_annotation(sentence: Sentence, annotation_type: str) -> bool:
    """Check if a sentence has a specific type of annotation."""
    for token in sentence.tokens:
        # Check token-level annotations
        if annotation_type == "lemma" and token.lemma and token.lemma != "_":
            return True
        if annotation_type == "xpos" and token.xpos and token.xpos != "_":
            return True
        if annotation_type == "upos" and token.upos and token.upos != "_":
            return True
        if annotation_type == "feats" and token.feats and token.feats != "_":
            return True
        if annotation_type == "head" and token.head > 0:
            return True
        if annotation_type == "deprel" and token.deprel and token.deprel != "_":
            return True
        if annotation_type == "normalization":
            # Check if any normalization attribute has a non-empty value
            # Note: empty string "" is the default, so we need to check for actual values
            reg = getattr(token, "reg", "")
            expan = getattr(token, "expan", "")
            mod = getattr(token, "mod", "")
            # Check if any of these have actual content (not empty, not "_", not "--")
            if reg and reg.strip() and reg not in ("_", "--", ""):
                return True
            if expan and expan.strip() and expan not in ("_", "--", ""):
                return True
            if mod and mod.strip() and mod not in ("_", "--", ""):
                return True
        
        # Check subtoken-level annotations
        for subtoken in token.subtokens:
            if annotation_type == "lemma" and subtoken.lemma and subtoken.lemma != "_":
                return True
            if annotation_type == "xpos" and subtoken.xpos and subtoken.xpos != "_":
                return True
            if annotation_type == "upos" and subtoken.upos and subtoken.upos != "_":
                return True
            if annotation_type == "feats" and subtoken.feats and subtoken.feats != "_":
                return True
            if annotation_type == "normalization":
                # Check if any normalization attribute has a non-empty value
                # Note: empty string "" is the default, so we need to check for actual values
                reg = getattr(subtoken, "reg", "")
                expan = getattr(subtoken, "expan", "")
                mod = getattr(subtoken, "mod", "")
                # Check if any of these have actual content (not empty, not "_", not "--")
                if reg and reg.strip() and reg not in ("_", "--", ""):
                    return True
                if expan and expan.strip() and expan not in ("_", "--", ""):
                    return True
                if mod and mod.strip() and mod not in ("_", "--", ""):
                    return True
    
    return False


def _sentence_has_required_annotations(
    sentence: Sentence,
    required_annotations: List[str],
    backend_type: str = "flexitag"
) -> bool:
    """
    Check if a sentence has the required annotations for training.
    
    This checks if at least one token in the sentence has each of the required
    annotation types. This partial requirement applies to all backends, though
    some backends may not train properly if annotations are incomplete.
    
    Args:
        sentence: Sentence to check
        required_annotations: List of required annotation types (e.g., ["xpos", "lemma", "head", "deprel"])
        backend_type: Type of backend ("flexitag" or neural backend name)
    
    Returns:
        True if sentence has all required annotations (at least one token has each type)
    """
    # For all backends, check if at least one token has each required annotation type
    # This is a partial requirement - we don't require all tokens to have all annotations
    return all(_has_annotation(sentence, ann) for ann in required_annotations)


def _collect_teitok_sentences(
    teitok_dir: Path,
    required_annotations: List[str],
    backend_type: str = "flexitag",
    verbose: bool = False,
    *,
    xpos_attr: Optional[str] = None,
    reg_attr: Optional[str] = None,
    expan_attr: Optional[str] = None,
    lemma_attr: Optional[str] = None,
    collect_all: bool = False,
) -> List[Sentence]:
    """
    Collect all sentences from TEITOK XML files in a directory.
    
    Args:
        teitok_dir: Directory containing TEITOK XML files
        required_annotations: List of required annotation types
        backend_type: Type of backend ("flexitag" or neural backend name)
        verbose: Whether to print progress
    
    Returns:
        List of sentences that have the required annotations
    """
    teitok_files = []
    for ext in ("*.xml", "*.tei", "*.teitok"):
        try:
            # Try both non-recursive and recursive search
            teitok_files.extend(teitok_dir.glob(ext))
            teitok_files.extend(teitok_dir.rglob(ext))
        except (PermissionError, OSError) as e:
            # If we can't access subdirectories, continue with what we found
            pass
    
    if not teitok_files:
        raise FileNotFoundError(
            f"No TEITOK XML files found in {teitok_dir}. "
            f"Expected files with extensions: .xml, .tei, .teitok. "
            f"Note: If files exist but are not accessible, check directory permissions."
        )
    
    all_sentences: List[Sentence] = []
    skipped_files = 0
    skipped_sentences = 0
    
    for teitok_file in teitok_files:
        try:
            doc = _load_teitok(
                teitok_file,
                xpos_attr=xpos_attr,
                reg_attr=reg_attr,
                expan_attr=expan_attr,
                lemma_attr=lemma_attr,
            )
            file_sentences = 0
            # Get filename without extension for sent_id prefix
            file_stem = teitok_file.stem
            file_sentence_counter = 0
            for sentence in doc.sentences:
                if collect_all or _sentence_has_required_annotations(sentence, required_annotations, backend_type):
                    file_sentence_counter += 1
                    # Update sent_id to include filename: filename-original_sent_id
                    original_sent_id = sentence.sent_id or sentence.id or sentence.source_id or ""
                    if original_sent_id:
                        # Combine filename with original sent_id
                        new_sent_id = f"{file_stem}-{original_sent_id}"
                    else:
                        # If no original sent_id, use filename with sentence index within file
                        new_sent_id = f"{file_stem}-s{file_sentence_counter}"
                    sentence.sent_id = new_sent_id
                    if not sentence.id:
                        sentence.id = new_sent_id
                    if not sentence.source_id:
                        sentence.source_id = original_sent_id or new_sent_id
                    all_sentences.append(sentence)
                    file_sentences += 1
                else:
                    skipped_sentences += 1
            if verbose and file_sentences == 0:
                print(f"[flexipipe] Skipped {teitok_file.name}: no sentences with required annotations")
                skipped_files += 1
        except Exception as e:
            if verbose:
                print(f"[flexipipe] Warning: Failed to load {teitok_file.name}: {e}")
            skipped_files += 1
    
    if not collect_all and not all_sentences:
        raise ValueError(
            f"No sentences with required annotations ({', '.join(required_annotations)}) "
            f"found in TEITOK files in {teitok_dir}"
        )
    
    return all_sentences


def _split_sentences(
    sentences: List[Sentence],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, List[Sentence]]:
    """
    Split sentences into train/dev/test sets.
    
    If output_dir is provided and existing split files exist, preserves existing splits
    based on sent_id to avoid contaminating test sets when rerunning.
    
    Args:
        sentences: List of sentences to split
        train_ratio: Proportion for training set (default: 0.8)
        dev_ratio: Proportion for dev set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        seed: Random seed for reproducibility
        output_dir: Optional output directory to check for existing splits
    
    Returns:
        Dictionary with "train", "dev", "test" keys containing lists of sentences
    """
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + dev_ratio + test_ratio}")
    
    # Check for existing splits if output_dir is provided
    existing_splits: Dict[str, set[str]] = {}
    if output_dir and output_dir.exists():
        for split_name in ("train", "dev", "test"):
            split_file = output_dir / f"{split_name}.conllu"
            if split_file.exists():
                try:
                    # Load existing split and extract sent_ids
                    existing_doc = conllu_to_document(split_file.read_text(encoding="utf-8"))
                    existing_sent_ids = {sent.sent_id or sent.id or "" for sent in existing_doc.sentences if sent.sent_id or sent.id}
                    if existing_sent_ids:
                        existing_splits[split_name] = existing_sent_ids
                except Exception:
                    # If we can't read the existing file, ignore it
                    pass
    
    # If we have existing splits, try to preserve them
    if existing_splits:
        # Create mapping from sent_id to sentence
        sent_id_to_sentence: Dict[str, Sentence] = {}
        for sent in sentences:
            sent_id = sent.sent_id or sent.id or ""
            if sent_id:
                sent_id_to_sentence[sent_id] = sent
        
        # Assign sentences to splits based on existing sent_ids
        result: Dict[str, List[Sentence]] = {"train": [], "dev": [], "test": []}
        unassigned: List[Sentence] = []
        
        for sent in sentences:
            sent_id = sent.sent_id or sent.id or ""
            assigned = False
            # Check each split in order (test first to preserve test set, then dev, then train)
            for split_name in ("test", "dev", "train"):
                if split_name in existing_splits and sent_id in existing_splits[split_name]:
                    result[split_name].append(sent)
                    assigned = True
                    break
            if not assigned:
                unassigned.append(sent)
        
        # If there are unassigned sentences, split them according to ratios
        if unassigned:
            if seed is not None:
                random.seed(seed)
            shuffled_unassigned = list(unassigned)
            random.shuffle(shuffled_unassigned)
            
            total_unassigned = len(shuffled_unassigned)
            train_end = int(total_unassigned * train_ratio)
            dev_end = train_end + int(total_unassigned * dev_ratio)
            
            result["train"].extend(shuffled_unassigned[:train_end])
            if dev_ratio > 0:
                result["dev"].extend(shuffled_unassigned[train_end:dev_end])
            if test_ratio > 0:
                result["test"].extend(shuffled_unassigned[dev_end:])
        
        return result
    
    # No existing splits, do normal random split
    if seed is not None:
        random.seed(seed)
    
    shuffled = list(sentences)
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    return {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end] if dev_ratio > 0 else [],
        "test": shuffled[dev_end:] if test_ratio > 0 else [],
    }


def _prepare_teitok_corpus(
    teitok_dir: Path,
    output_dir: Path,
    required_annotations: List[str],
    backend_type: str = "flexitag",
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
    verbose: bool = False,
    *,
    xpos_attr: Optional[str] = None,
    reg_attr: Optional[str] = None,
    expan_attr: Optional[str] = None,
    lemma_attr: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Prepare a TEITOK corpus or CoNLL-U treebank for training by converting to CoNLL-U and splitting.
    
    If input is already CoNLL-U (file or directory with .conllu files), loads those instead.
    Handles treebanks with missing sections (e.g., only test, or only train+test).
    
    Args:
        teitok_dir: Directory containing TEITOK XML files, or CoNLL-U file/directory
        output_dir: Directory to write CoNLL-U files
        required_annotations: List of required annotation types
        backend_type: Type of backend ("flexitag" or neural backend name)
        train_ratio: Proportion for training set
        dev_ratio: Proportion for dev set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        verbose: Whether to print progress
    
    Returns:
        Dictionary mapping split names ("train", "dev", "test") to Path objects
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result dictionary early (needed for incomplete sentences)
    result: Dict[str, Path] = {}
    
    # Check if input is already CoNLL-U (file or directory with .conllu files)
    sentences: List[Sentence] = []
    is_conllu_input = False
    
    if teitok_dir.is_file() and teitok_dir.suffix.lower() in (".conllu", ".conll"):
        # Single CoNLL-U file
        is_conllu_input = True
        if verbose:
            print(f"[flexipipe] Loading CoNLL-U file: {teitok_dir.name}")
        doc = conllu_to_document(teitok_dir.read_text(encoding="utf-8"))
        sentences = doc.sentences
        # Ensure sent_ids include filename for tracking
        file_stem = teitok_dir.stem
        for idx, sent in enumerate(sentences):
            if not sent.sent_id or not sent.sent_id.startswith(file_stem):
                original_sent_id = sent.sent_id or sent.id or sent.source_id or ""
                if original_sent_id:
                    sent.sent_id = f"{file_stem}-{original_sent_id}"
                else:
                    sent.sent_id = f"{file_stem}-s{idx + 1}"
                if not sent.id:
                    sent.id = sent.sent_id
    elif teitok_dir.is_dir():
        # Check if directory contains CoNLL-U files
        conllu_files = list(teitok_dir.glob("*.conllu")) + list(teitok_dir.glob("*.conll"))
        if conllu_files:
            is_conllu_input = True
            if verbose:
                print(f"[flexipipe] Loading CoNLL-U files from directory: {len(conllu_files)} files")
            for conllu_file in sorted(conllu_files):
                doc = conllu_to_document(conllu_file.read_text(encoding="utf-8"))
                file_stem = conllu_file.stem
                for idx, sent in enumerate(doc.sentences):
                    # Ensure sent_ids include filename for tracking
                    if not sent.sent_id or not sent.sent_id.startswith(file_stem):
                        original_sent_id = sent.sent_id or sent.id or sent.source_id or ""
                        if original_sent_id:
                            sent.sent_id = f"{file_stem}-{original_sent_id}"
                        else:
                            sent.sent_id = f"{file_stem}-s{idx + 1}"
                        if not sent.id:
                            sent.id = sent.sent_id
                sentences.extend(doc.sentences)
    
    # If we loaded CoNLL-U, skip TEITOK processing
    if is_conllu_input:
        total_tokens = sum(len(sent.tokens) for sent in sentences)
        print(f"[flexipipe] Loaded {len(sentences):,} sentences ({total_tokens:,} tokens) from CoNLL-U input")
        
        # Check if input directory already has split files - if so, preserve those splits
        if teitok_dir.is_dir():
            existing_input_splits: Dict[str, set[str]] = {}
            for split_name in ("train", "dev", "test"):
                split_file = teitok_dir / f"{split_name}.conllu"
                if split_file.exists():
                    try:
                        existing_doc = conllu_to_document(split_file.read_text(encoding="utf-8"))
                        existing_sent_ids = {sent.sent_id or sent.id or "" for sent in existing_doc.sentences if sent.sent_id or sent.id}
                        if existing_sent_ids:
                            existing_input_splits[split_name] = existing_sent_ids
                            if verbose:
                                print(f"[flexipipe] Found existing {split_name} split in input: {len(existing_sent_ids)} sentences")
                    except Exception:
                        pass
            
            # If we found existing splits in input, use those instead of random split
            if existing_input_splits:
                # Reorganize sentences according to existing splits
                sent_id_to_sentence: Dict[str, Sentence] = {}
                for sent in sentences:
                    sent_id = sent.sent_id or sent.id or ""
                    if sent_id:
                        sent_id_to_sentence[sent_id] = sent
                
                split_sentences: Dict[str, List[Sentence]] = {"train": [], "dev": [], "test": []}
                unassigned: List[Sentence] = []
                
                for sent in sentences:
                    sent_id = sent.sent_id or sent.id or ""
                    assigned = False
                    for split_name in ("test", "dev", "train"):
                        if split_name in existing_input_splits and sent_id in existing_input_splits[split_name]:
                            split_sentences[split_name].append(sent)
                            assigned = True
                            break
                    if not assigned:
                        unassigned.append(sent)
                
                # If there are unassigned sentences, split them according to ratios
                if unassigned:
                    if seed is not None:
                        random.seed(seed)
                    shuffled_unassigned = list(unassigned)
                    random.shuffle(shuffled_unassigned)
                    
                    total_unassigned = len(shuffled_unassigned)
                    train_end = int(total_unassigned * train_ratio)
                    dev_end = train_end + int(total_unassigned * dev_ratio)
                    
                    split_sentences["train"].extend(shuffled_unassigned[:train_end])
                    if dev_ratio > 0:
                        split_sentences["dev"].extend(shuffled_unassigned[train_end:dev_end])
                    if test_ratio > 0:
                        split_sentences["test"].extend(shuffled_unassigned[dev_end:])
                    
                    if verbose and unassigned:
                        print(f"[flexipipe] Split {len(unassigned)} unassigned sentences according to ratios")
                
                # Write splits to output directory
                split_token_counts: Dict[str, int] = {}
                for split_name, split_sents in split_sentences.items():
                    token_count = sum(len(sent.tokens) for sent in split_sents) if split_sents else 0
                    split_token_counts[split_name] = token_count
                    
                    if not split_sents:
                        continue
                    
                    split_doc = Document(id=f"{split_name}_split")
                    split_doc.sentences = split_sents
                    output_file = output_dir / f"{split_name}.conllu"
                    conllu_text = document_to_conllu(split_doc, create_implicit_mwt=False)
                    output_file.write_text(conllu_text, encoding="utf-8")
                    result[split_name] = output_file
                    
                    if verbose:
                        print(f"[flexipipe] Wrote {len(split_sents)} sentences to {output_file}")
                
                result["_token_counts"] = split_token_counts  # type: ignore
                return result
    else:
        # Collect all sentences from TEITOK
        # For flexitag, we collect both complete and incomplete sentences
        # For neural backends, we only collect complete sentences
        if backend_type == "flexitag":
            # Collect all sentences (both complete and incomplete)
            all_sentences = _collect_teitok_sentences(
                teitok_dir,
                required_annotations,
                backend_type,
                verbose,
                xpos_attr=xpos_attr,
                reg_attr=reg_attr,
                expan_attr=expan_attr,
                lemma_attr=lemma_attr,
                collect_all=True,  # Collect all sentences, not just complete ones
            )
            
            # Report total processed
            total_sentences = len(all_sentences)
            total_tokens = sum(len(sent.tokens) for sent in all_sentences)
            print(f"[flexipipe] Processed {total_sentences:,} sentences ({total_tokens:,} tokens) from TEITOK corpus")
            
            # Separate complete and incomplete sentences
            complete_sentences = []
            incomplete_sentences = []
            for sentence in all_sentences:
                if _sentence_has_required_annotations(sentence, required_annotations, backend_type):
                    complete_sentences.append(sentence)
                else:
                    incomplete_sentences.append(sentence)
            
            # Report complete vs incomplete
            complete_tokens = sum(len(sent.tokens) for sent in complete_sentences)
            incomplete_tokens = sum(len(sent.tokens) for sent in incomplete_sentences)
            print(f"[flexipipe] Complete: {len(complete_sentences):,} sentences ({complete_tokens:,} tokens)")
            if incomplete_sentences:
                print(f"[flexipipe] Incomplete: {len(incomplete_sentences):,} sentences ({incomplete_tokens:,} tokens)")
            
            # Write incomplete sentences to incomplete.conllu if any
            if incomplete_sentences:
                incomplete_doc = Document(id="incomplete")
                incomplete_doc.sentences = incomplete_sentences
                incomplete_file = output_dir / "incomplete.conllu"
                incomplete_text = document_to_conllu(incomplete_doc, create_implicit_mwt=False)
                incomplete_file.write_text(incomplete_text, encoding="utf-8")
                result["incomplete"] = incomplete_file
                print(f"[flexipipe] Wrote incomplete sentences to {incomplete_file}")
            
            sentences = complete_sentences
        else:
            # For neural backends, only collect complete sentences
            sentences = _collect_teitok_sentences(
                teitok_dir,
                required_annotations,
                backend_type,
                verbose,
                xpos_attr=xpos_attr,
                reg_attr=reg_attr,
                expan_attr=expan_attr,
                lemma_attr=lemma_attr,
            )
            # Report for neural backends
            total_tokens = sum(len(sent.tokens) for sent in sentences)
            print(f"[flexipipe] Processed {len(sentences):,} complete sentences ({total_tokens:,} tokens) from TEITOK corpus")
    
    # Split sentences (pass output_dir to preserve existing splits if rerunning)
    # Note: If input was CoNLL-U with existing splits, they will be preserved
    splits = _split_sentences(sentences, train_ratio, dev_ratio, test_ratio, seed, output_dir=output_dir)
    
    # Write CoNLL-U files and collect token counts
    split_token_counts: Dict[str, int] = {}
    for split_name, split_sentences in splits.items():
        # Record token count (0 for empty splits)
        token_count = sum(len(sent.tokens) for sent in split_sentences) if split_sentences else 0
        split_token_counts[split_name] = token_count
        
        if not split_sentences:
            # Skip writing empty splits, but ensure the key exists in result for error checking
            continue
        
        # Create a temporary document with these sentences
        split_doc = Document(id=f"{split_name}_split")
        split_doc.sentences = split_sentences
        
        # Write to CoNLL-U
        output_file = output_dir / f"{split_name}.conllu"
        conllu_text = document_to_conllu(split_doc, create_implicit_mwt=False)
        output_file.write_text(conllu_text, encoding="utf-8")
        result[split_name] = output_file
        
        if verbose:
            print(f"[flexipipe] Wrote {len(split_sentences)} sentences to {output_file}")
    
    # Store token counts in result for summary display
    result["_token_counts"] = split_token_counts  # type: ignore
    
    # Ensure "train" key exists in result (even if empty) for error checking upstream
    if "train" not in result and "train" in splits:
        # Train split exists but is empty - this will be caught by the check in train_ud_treebank
        pass
    
    return result


def _find_ud_splits(ud_root: Path) -> Dict[str, Path]:
    """Find train/dev/test splits, supporting both CoNLL-U and TEITOK formats."""
    splits = {}
    # Try CoNLL-U files first (standard UD format)
    for split in ("train", "dev", "test"):
        # Try standard UD naming: *-ud-{split}.conllu
        matches = list(ud_root.glob(f"*ud-{split}.conllu"))
        if not matches:
            # Try alternative naming: *-{split}.conllu
            matches = list(ud_root.glob(f"*-{split}.conllu"))
        if not matches:
            # Try files named exactly {split}.conllu
            matches = list(ud_root.glob(f"{split}.conllu"))
        if not matches:
            # Try recursive search
            matches = list(ud_root.glob(f"**/*ud-{split}.conllu"))
        if not matches:
            matches = list(ud_root.glob(f"**/*-{split}.conllu"))
        if not matches:
            matches = list(ud_root.glob(f"**/{split}.conllu"))
        if matches:
            splits[split] = matches[0]
            continue
        
        # Try TEITOK XML files
        teitok_matches = list(ud_root.glob(f"*ud-{split}.xml"))
        if not teitok_matches:
            teitok_matches = list(ud_root.glob(f"*-{split}.xml"))
        if not teitok_matches:
            teitok_matches = list(ud_root.glob(f"*ud-{split}.tei"))
        if not teitok_matches:
            teitok_matches = list(ud_root.glob(f"*-{split}.tei"))
        if not teitok_matches:
            # Try recursive search
            teitok_matches = list(ud_root.glob(f"**/*ud-{split}.xml"))
        if not teitok_matches:
            teitok_matches = list(ud_root.glob(f"**/*-{split}.xml"))
        if not teitok_matches:
            teitok_matches = list(ud_root.glob(f"**/*ud-{split}.tei"))
        if not teitok_matches:
            teitok_matches = list(ud_root.glob(f"**/*-{split}.tei"))
        if teitok_matches:
            splits[split] = teitok_matches[0]
    
    # If no pre-split files found, check if this is a TEITOK corpus directory
    if "train" not in splits:
        # Check if directory contains TEITOK XML files (recursive search)
        teitok_files = []
        for ext in ("*.xml", "*.tei", "*.teitok"):
            # Try both non-recursive and recursive
            try:
                teitok_files.extend(ud_root.glob(ext))
                teitok_files.extend(ud_root.rglob(ext))  # Use rglob for recursive search
            except (PermissionError, OSError):
                # If we can't access the directory, assume it's a TEITOK corpus
                # and let _collect_teitok_sentences handle it
                pass
        
        if teitok_files:
            # This is a TEITOK corpus - we'll handle it specially
            return {}  # Signal that this needs special handling
    
    if "train" not in splits:
        raise FileNotFoundError(
            f"Could not locate training data file underneath {ud_root}. "
            f"Expected files matching patterns: *-ud-train.conllu, *-train.conllu, "
            f"*-ud-train.xml, *-train.xml, *-ud-train.tei, or *-train.tei, "
            f"or a directory of TEITOK XML files"
        )
    return splits


def _build_model_payload(
    documents: List[Document],
    splits: Dict[str, Path],
    ud_root: Path,
    model_name: Optional[str],
    tag_attribute: str,
    include_dev: bool = False,
) -> Dict[str, Any]:
    tag_attribute = tag_attribute.lower()

    def _clean(value: str) -> str:
        return "" if not value or value in {"_", "-"} else value

    def _compose_tag(xpos: str, upos: str, feats: str) -> str:
        xpos_clean = _clean(xpos)
        upos_clean = _clean(upos)
        feats_clean = _clean(feats)
        if tag_attribute == "upos":
            return upos_clean
        if tag_attribute == "utot":
            if upos_clean and feats_clean:
                return f"{upos_clean}#{feats_clean}"
            return upos_clean
        return xpos_clean

    def _token_tag_value(token: Token) -> str:
        return _compose_tag(token.xpos, token.upos, token.feats)

    def _subtoken_tag_value(sub: SubToken) -> str:
        feats_value = getattr(sub, "feats", "") or ""
        return _compose_tag(sub.xpos, sub.upos, feats_value)

    vocab: Dict[str, Dict[Tuple[str, str, str, str, str, str], Counter]] = defaultdict(lambda: defaultdict(Counter))
    multiword_parts: Dict[str, Counter] = defaultdict(Counter)
    multiword_parts_tags: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, ...]] = {}
    multiword_parts_full: Dict[Tuple[str, Tuple[str, ...]], List[Dict]] = {}  # Store full part objects
    capitalizable: Dict[str, Counter] = defaultdict(Counter)
    transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    start_transitions: Dict[str, int] = defaultdict(int)

    total_sentences = 0
    total_tokens = 0

    for doc in documents:
        for sentence in doc.sentences:
            total_sentences += 1
            atomic_tokens = list(_iter_atomic_tokens(sentence))
            for idx, tok in enumerate(atomic_tokens):
                total_tokens += 1
                tag_val = _token_tag_value(tok)
                if tag_val:
                    case = _form_case(tok.form)
                    capitalizable[tag_val][case] += 1
                    if idx == 0:
                        start_transitions[tag_val] += 1
                    else:
                        prev_tag = _token_tag_value(atomic_tokens[idx - 1])
                        if prev_tag:
                            transitions[prev_tag][tag_val] += 1

            for token in sentence.tokens:
                if token.is_mwt and token.subtokens:
                    parts = tuple(sub.form for sub in token.subtokens)
                    multiword_parts[token.form][parts] += 1
                    part_tags = tuple(_subtoken_tag_value(sub) for sub in token.subtokens)
                    multiword_parts_tags[(token.form, parts)] = part_tags
                    # Store full part information
                    part_objects = []
                    for sub in token.subtokens:
                        part_obj = {
                            "form": sub.form,
                            "lemma": sub.lemma or sub.form,
                            "upos": sub.upos or "_",
                            "xpos": sub.xpos or "_",
                            "feats": getattr(sub, "feats", "") or "_",
                        }
                        if hasattr(sub, "reg") and sub.reg:
                            part_obj["reg"] = sub.reg
                        if hasattr(sub, "expan") and sub.expan:
                            part_obj["expan"] = sub.expan
                        part_objects.append(part_obj)
                    # Store contraction-level attributes (reg, expan, mod, trslit, ltrslit, tokid)
                    # These are at the parent token level, not the parts level
                    contraction_attrs = {}
                    if hasattr(token, "reg") and token.reg:
                        contraction_attrs["reg"] = token.reg
                    if hasattr(token, "expan") and token.expan:
                        contraction_attrs["expan"] = token.expan
                    if hasattr(token, "mod") and token.mod:
                        contraction_attrs["mod"] = token.mod
                    if hasattr(token, "trslit") and token.trslit:
                        contraction_attrs["trslit"] = token.trslit
                    if hasattr(token, "ltrslit") and token.ltrslit:
                        contraction_attrs["ltrslit"] = token.ltrslit
                    if hasattr(token, "tokid") and token.tokid:
                        contraction_attrs["tokid"] = token.tokid
                    if contraction_attrs:
                        multiword_parts_full[(token.form, parts)] = (part_objects, contraction_attrs)
                    else:
                        multiword_parts_full[(token.form, parts)] = part_objects
                    for sub in token.subtokens:
                        sub_key = (
                            sub.lemma or "",
                            sub.xpos or "",
                            sub.upos or "",
                            getattr(sub, "feats", "") or "",
                            getattr(sub, "reg", "") or "",
                            getattr(sub, "expan", "") or "",
                        )
                        vocab[sub.form][sub_key]["count"] += 1
                    continue

                key = (
                    token.lemma or "",
                    token.xpos or "",
                    token.upos or "",
                    token.feats or "",
                    token.reg or "",
                    token.expan or "",
                )
                vocab[token.form][key]["count"] += 1

    # Only include files actually used for training (train and optionally dev)
    # Do NOT include test files in source_files to avoid confusion
    training_files = [splits["train"]]
    if "dev" in splits and include_dev:
        training_files.append(splits["dev"])
    
    metadata = {
        "corpus_name": model_name or ud_root.name,
        "creation_date": datetime.utcnow().isoformat(timespec="seconds"),
        "source_folders": [str(ud_root)],
        "source_files": [str(p) for p in training_files],
        "tag_attribute": tag_attribute,
        "language": None,
        "capitalizable_tags": {
            tag_attribute: {tag: dict(counts) for tag, counts in capitalizable.items()},
        },
        "vocab_stats": {
            "total_entries": sum(len(analyses) for analyses in vocab.values()) + sum(len(parts) for parts in multiword_parts.values()),
            "word_entries": len(vocab),
            "multiword_entries": len(multiword_parts),
            "total_sentences": total_sentences,
            "total_tokens": total_tokens,
        },
    }

    vocab_json = {}
    for form, analyses in vocab.items():
        entries = []
        for (lemma, xpos, upos, feats, reg, expan), counters in analyses.items():
            tag_value = _compose_tag(xpos, upos, feats) or "_"
            entry = {
                "lemma": lemma or "_",
                "xpos": xpos or "_",
                "upos": upos or "_",
                "feats": feats or "_",
                "count": counters["count"],
                "tag": tag_value,
            }
            if reg:
                entry["reg"] = reg
            if expan:
                entry["expan"] = expan
            entries.append(entry)
        if len(entries) == 1:
            vocab_json[form] = entries[0]
        else:
            vocab_json[form] = entries

    for form, parts_counter in multiword_parts.items():
        entries = []
        for parts, count in parts_counter.items():
            entry = {
                "lemma": "_",
                "count": count,
            }
            # Store full part objects instead of just strings and tags
            part_data = multiword_parts_full.get((form, parts))
            if part_data:
                # Check if it's a tuple (part_objects, contraction_attrs) or just part_objects
                if isinstance(part_data, tuple) and len(part_data) == 2:
                    part_objects, contraction_attrs = part_data
                    entry["parts"] = part_objects
                    # Store contraction-level attributes
                    if contraction_attrs:
                        for attr_name in ["reg", "expan", "mod", "trslit", "ltrslit", "tokid"]:
                            if attr_name in contraction_attrs:
                                entry[attr_name] = contraction_attrs[attr_name]
                else:
                    # Just part_objects (list)
                    entry["parts"] = part_data
            else:
                # Fallback: use old format if full objects not available
                entry["parts"] = list(parts)
                part_tags = multiword_parts_tags.get((form, parts))
                if part_tags and any(part_tags):
                    entry["parts_tags"] = list(part_tags)
            entries.append(entry)
        if form in vocab_json:
            existing = vocab_json[form]
            if isinstance(existing, list):
                existing.extend(entries)
            else:
                vocab_json[form] = [existing, *entries]
        else:
            vocab_json[form] = entries[0] if len(entries) == 1 else entries

    def _condense_transitions(table: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        condensed: Dict[str, Dict[str, int]] = {}
        for prev_tag, next_counts in table.items():
            if not next_counts:
                continue
            condensed[prev_tag] = dict(next_counts)
        return condensed

    transitions_json = {
        tag_attribute: _condense_transitions(transitions),
        "start": dict(start_transitions),
    }

    # Default tagger settings (can be overridden via CLI arguments or fine-tuning)
    default_settings: Dict[str, Any] = dict(DEFAULT_TAGGER_SETTINGS)
    default_settings["tagpos"] = tag_attribute

    payload = {
        "metadata": metadata,
        "vocab": vocab_json,
        "transitions": transitions_json,
        "settings": default_settings,
    }

    return payload


def train_ud_treebank(
    ud_root: Path,
    output_dir: Path,
    model_name: str | None = None,
    include_dev: bool = False,
    verbose: bool = False,
    finetune: str = "none",
    tag_attribute: Optional[str] = None,
    language_code: Optional[str] = None,
    ud_folder: Optional[str] = None,
    *,
    nlpform: str = "form",
    xpos_attr: Optional[str] = None,
    reg_attr: Optional[str] = None,
    expan_attr: Optional[str] = None,
    lemma_attr: Optional[str] = None,
) -> Path:
    tag_attribute_input = tag_attribute.lower() if tag_attribute else None
    if tag_attribute_input not in {None, "auto", "xpos", "upos", "utot"}:
        raise ValueError(
            f"Unsupported tag attribute '{tag_attribute}'. Choose from xpos, upos, utot, or auto."
        )

    auto_select = tag_attribute_input in (None, "auto")
    candidate_tag_attributes = ["xpos", "upos", "utot"] if auto_select else [tag_attribute_input]

    ud_root = ud_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _find_ud_splits(ud_root)
    
    # Handle TEITOK corpus directory (no pre-split files)
    teitok_temp_dir = None
    if not splits or "train" not in splits:
        # This is a TEITOK corpus - prepare it
        # Use ud_folder if provided, otherwise create a temporary directory
        if ud_folder:
            tmp_path = Path(ud_folder).expanduser().resolve()
            tmp_path.mkdir(parents=True, exist_ok=True)
            teitok_temp_dir = str(tmp_path)
        else:
            teitok_temp_dir = tempfile.mkdtemp(prefix="flexipipe-teitok-")
            tmp_path = Path(teitok_temp_dir)
        # For flexitag, we need lemma or xpos
        # If xpos_attr is specified, we require xpos (it's the tag attribute we'll use)
        # If lemma_attr is specified, we also require lemma
        # Otherwise, we require lemma by default
        required_annotations = []
        if xpos_attr:
            required_annotations.append("xpos")
        if lemma_attr:
            required_annotations.append("lemma")
        if not required_annotations:
            # Default: require lemma if nothing specified
            required_annotations = ["lemma"]
        try:
            prepared_splits = _prepare_teitok_corpus(
                teitok_dir=ud_root,
                output_dir=tmp_path,
                required_annotations=required_annotations,
                backend_type="flexitag",
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                seed=42,  # Fixed seed for reproducibility
                verbose=verbose,
                xpos_attr=xpos_attr,
                reg_attr=reg_attr,
                expan_attr=expan_attr,
                lemma_attr=lemma_attr,
            )
            splits = prepared_splits
            
            # Check if we have a train split (required for training)
            # Note: prepared_splits may have _token_counts but not train if all sentences were filtered
            if "train" not in splits or (isinstance(splits.get("train"), Path) and not splits["train"].exists()):
                # Check token counts to see if any sentences were found
                split_counts = prepared_splits.get("_token_counts", {})
                total_tokens = sum(split_counts.values()) if split_counts else 0
                if total_tokens == 0:
                    raise ValueError(
                        "No training data available. All sentences were filtered out as incomplete. "
                        f"Check that your TEITOK files contain the required annotations: {required_annotations}"
                    )
                else:
                    raise ValueError(
                        "No training split available. All sentences were assigned to dev/test splits. "
                        f"Found {total_tokens:,} tokens total across all splits."
                    )
            
            # If ud_folder is provided, print where the files are kept
            if ud_folder:
                print(f"[flexipipe] CoNLL-U files saved to: {teitok_temp_dir}")
                if "test" in prepared_splits:
                    print(f"[flexipipe] Test file available at: {prepared_splits['test']}")
                
                # Print token distribution summary
                split_counts = prepared_splits.get("_token_counts", {})
                if split_counts:
                    total_tokens = sum(split_counts.values())
                    if total_tokens > 0:
                        parts = []
                        for split_name in ["train", "dev", "test"]:
                            if split_name in split_counts:
                                count = split_counts[split_name]
                                pct = (count / total_tokens) * 100
                                parts.append(f"{split_name} = {count:,} tokens ({pct:.1f}%)")
                        print(f"[flexipipe] Created gold standard distribution: {', '.join(parts)}")
                # Remove the token counts from the result dict
                prepared_splits.pop("_token_counts", None)
        except Exception as e:
            import shutil
            # Only clean up if it's a temporary directory (not ud_folder)
            if teitok_temp_dir and not ud_folder:
                shutil.rmtree(teitok_temp_dir, ignore_errors=True)
            raise
    
    # Final check: ensure we have a train split before proceeding
    if "train" not in splits:
        raise ValueError(
            "No training data available. All sentences were filtered out as incomplete. "
            "Check that your TEITOK files contain the required annotations."
        )
    
    train_doc = _load_training_file(splits["train"])
    if nlpform and nlpform != "form":
        train_doc = apply_nlpform(train_doc, nlpform)
    documents: List[Document] = [train_doc]

    dev_doc: Optional[Document] = None
    if "dev" in splits:
        dev_doc = _load_training_file(splits["dev"])
        if nlpform and nlpform != "form":
            dev_doc = apply_nlpform(dev_doc, nlpform)
        if include_dev and dev_doc is not None:
            documents.append(dev_doc)

    # CRITICAL: Never evaluate on training data - it leads to inflated accuracy
    # If no dev set exists, we cannot do proper evaluation during training
    # In this case, we skip evaluation and only report that dev was unavailable
    evaluation_doc = dev_doc
    evaluation_set_name = "dev" if dev_doc is not None else None

    candidate_results: List[Dict[str, Any]] = []
    if auto_select and len(candidate_tag_attributes) > 1:
        if evaluation_doc is not None:
            print(f"[flexipipe] evaluating {len(candidate_tag_attributes)} tag attribute candidates on {evaluation_set_name} set...")
        else:
            print(f"[flexipipe] WARNING: no dev set found, cannot evaluate tag attributes. Using first candidate: {candidate_tag_attributes[0]}")
    
    for idx, attr in enumerate(candidate_tag_attributes, 1):
        if auto_select and len(candidate_tag_attributes) > 1:
            print(f"[flexipipe] testing {attr} ({idx}/{len(candidate_tag_attributes)})...", end=" ", flush=True)
        
        payload = _build_model_payload(documents, splits, ud_root, model_name, attr, include_dev)
        baseline: Optional[CandidateResult] = None
        error: Optional[str] = None
        
        # Only evaluate if we have a dev set (never evaluate on training data)
        if evaluation_doc is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=output_dir) as tmp:
                tmp_file = Path(tmp.name)
            try:
                _write_model(tmp_file, payload)
                try:
                    baseline = _evaluate_candidate(tmp_file, evaluation_doc, payload["settings"])
                    if baseline is None:
                        error = "evaluation returned no metrics"
                        if auto_select and len(candidate_tag_attributes) > 1:
                            print(f"error: {error} (skipping {attr})")
                    elif auto_select and len(candidate_tag_attributes) > 1:
                        print(
                            f"pos={baseline.pos_accuracy*100:.2f}% "
                            f"lemma={baseline.lemma_accuracy*100:.2f}% "
                            f"speed={baseline.speed:.0f} tok/s"
                        )
                except (RuntimeError, UnicodeDecodeError) as exc:
                    error = str(exc)
                    if auto_select and len(candidate_tag_attributes) > 1:
                        print(f"error: {error} (skipping {attr})")
            finally:
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
        else:
            # No dev set - skip evaluation, use first candidate or default
            if auto_select and len(candidate_tag_attributes) > 1:
                print("(no dev set, skipping evaluation)")
        
        candidate_results.append({
            "tag_attribute": attr,
            "payload": payload,
            "baseline": baseline,
            "error": error,
        })
    
    if auto_select and len(candidate_tag_attributes) > 1:
        def _score_preview(result: Dict[str, Any]) -> float:
            baseline_result = result["baseline"]
            return baseline_result.pos_accuracy if baseline_result else float("-inf")
        best_result_preview = max(candidate_results, key=_score_preview)
        if best_result_preview["baseline"] is None:
            raise RuntimeError("No tag attribute candidate produced evaluation metrics; aborting training.")
        print(f"[flexipipe] selected {best_result_preview['tag_attribute']} (best POS accuracy)")

    def _score(result: Dict[str, Any]) -> float:
        baseline_result = result["baseline"]
        return baseline_result.pos_accuracy if baseline_result else float("-inf")

    best_result = max(candidate_results, key=_score)
    if best_result["baseline"] is None:
        raise RuntimeError("All tag attribute candidates failed to produce evaluation metrics; aborting training.")
    tag_attribute = best_result["tag_attribute"]
    payload: Dict[str, Any] = best_result["payload"]
    metadata = payload["metadata"]
    if language_code:
        metadata["language"] = language_code
    default_settings = payload["settings"]

    tag_selection_entries: List[Dict[str, Any]] = []
    for result in candidate_results:
        baseline = result["baseline"]
        tag_selection_entries.append({
            "tag_attribute": result["tag_attribute"],
            "pos_accuracy": baseline.pos_accuracy if baseline else None,
            "lemma_accuracy": baseline.lemma_accuracy if baseline else None,
            "speed": baseline.speed if baseline else None,
            "error": result["error"],
        })

    best_baseline = best_result["baseline"]
    metadata["tag_selection"] = {
        "candidates": tag_selection_entries,
        "chosen": tag_attribute,
        "evaluation_set": evaluation_set_name,
        "auto": auto_select,
    }
    if best_baseline is not None:
        metadata["tag_selection"]["chosen_baseline"] = {
            "pos_accuracy": best_baseline.pos_accuracy,
            "lemma_accuracy": best_baseline.lemma_accuracy,
            "speed": best_baseline.speed,
        }
    if best_result["error"]:
        metadata["tag_selection"]["chosen_error"] = best_result["error"]

    metadata["final_metrics"] = {
        "pos_accuracy": best_baseline.pos_accuracy if best_baseline else None,
        "lemma_accuracy": best_baseline.lemma_accuracy if best_baseline else None,
        "speed": best_baseline.speed if best_baseline else None,
    }

    # Evaluate different endlen values (0, positive, negative) to find the best direction
    # This happens after tag attribute selection but before fine-tuning
    # NOTE: The vocabulary JSON doesn't store endings - they are computed when the lexicon loads.
    # The endlen setting affects how endings are indexed (prefixes vs suffixes), so we can
    # use the same vocabulary JSON and just pass different endlen values in the options.
    endlen_candidates = [0, 3, -3]  # 0 = disabled, 3 = suffixes, -3 = prefixes
    best_endlen = default_settings.get("endlen", 6)  # Default fallback
    endlen_selection_entries: List[Dict[str, Any]] = []
    
    if evaluation_doc is not None and best_baseline is not None:
        if verbose:
            print(f"[flexipipe] evaluating {len(endlen_candidates)} endlen candidates on {evaluation_set_name} set...")
        
        # Create a temporary model file with the vocabulary (endings will be indexed on load)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=output_dir) as tmp:
            tmp_file = Path(tmp.name)
        
        try:
            _write_model(tmp_file, payload)
            
            for idx, endlen_val in enumerate(endlen_candidates, 1):
                if verbose:
                    print(f"[flexipipe] testing endlen={endlen_val} ({idx}/{len(endlen_candidates)})...", end=" ", flush=True)
                
                # Create settings with this endlen value
                # The endlen will be used when the lexicon loads to index endings correctly
                test_settings = default_settings.copy()
                test_settings["endlen"] = endlen_val
                
                baseline: Optional[CandidateResult] = None
                error: Optional[str] = None
                
                try:
                    baseline = _evaluate_candidate(tmp_file, evaluation_doc, test_settings)
                    if baseline is None:
                        error = "evaluation returned no metrics"
                        if verbose:
                            print(f"error: {error}")
                    elif verbose:
                        print(
                            f"pos={baseline.pos_accuracy*100:.2f}% "
                            f"lemma={baseline.lemma_accuracy*100:.2f}% "
                            f"speed={baseline.speed:.0f} tok/s"
                        )
                except (RuntimeError, UnicodeDecodeError) as exc:
                    error = str(exc)
                    if verbose:
                        print(f"error: {error}")
                
                endlen_selection_entries.append({
                    "endlen": endlen_val,
                    "pos_accuracy": baseline.pos_accuracy if baseline else None,
                    "lemma_accuracy": baseline.lemma_accuracy if baseline else None,
                    "speed": baseline.speed if baseline else None,
                    "error": error,
                })
            
            # Select best endlen based on POS accuracy
            def _score_endlen(result: Dict[str, Any]) -> float:
                return result["pos_accuracy"] if result["pos_accuracy"] is not None else float("-inf")
            
            best_endlen_result = max(endlen_selection_entries, key=_score_endlen)
            if best_endlen_result["pos_accuracy"] is not None:
                best_endlen = best_endlen_result["endlen"]
                if verbose:
                    print(f"[flexipipe] selected endlen={best_endlen} (best POS accuracy)")
            else:
                if verbose:
                    print(f"[flexipipe] WARNING: no endlen candidate produced metrics, using default endlen={best_endlen}")
        finally:
            try:
                tmp_file.unlink()
            except OSError:
                pass
    
    # Update default_settings with the selected endlen
    default_settings["endlen"] = best_endlen
    payload["settings"] = default_settings
    
    # Store endlen selection in metadata
    metadata["endlen_selection"] = {
        "candidates": endlen_selection_entries,
        "chosen": best_endlen,
        "evaluation_set": evaluation_set_name,
    }
    if endlen_selection_entries:
        best_endlen_baseline = max(endlen_selection_entries, key=lambda r: r["pos_accuracy"] if r["pos_accuracy"] is not None else float("-inf"))
        if best_endlen_baseline["pos_accuracy"] is not None:
            metadata["endlen_selection"]["chosen_baseline"] = {
                "pos_accuracy": best_endlen_baseline["pos_accuracy"],
                "lemma_accuracy": best_endlen_baseline["lemma_accuracy"],
                "speed": best_endlen_baseline["speed"],
            }

    output_path = output_dir / "model_vocab.json"

    finetune_mode = finetune.lower()
    if finetune_mode not in {"none", "accuracy", "speed", "balanced"}:
        raise ValueError("finetune must be one of: none, accuracy, speed, balanced")

    search_space = {param: list(values) for param, values in PARAM_SEARCH_SPACE.items()}
    search_space["tagpos"] = [tag_attribute]

    tuning_summary: Optional[Dict[str, Any]] = None
    if finetune_mode != "none":
        if best_baseline is None:
            metadata["tag_selection"]["fine_tune_skipped"] = "baseline_unavailable"
            if verbose:
                print(
                    "[flexipipe] skipping fine-tuning because baseline evaluation was unavailable for the selected tag attribute"
                )
            finetune_mode = "none"

    if finetune_mode != "none":
        if verbose:
            print(
                f"[flexipipe] fine-tuning on {evaluation_set_name} set ({len(evaluation_doc.sentences)} sentences)"
            )
        _write_model(output_path, payload)
        try:
            tuned_settings, summary = _finetune_settings(
                output_path,
                evaluation_doc,
                default_settings,
                finetune_mode,
                search_space,
                verbose=verbose,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Fine-tuning failed: {exc}") from exc
        payload["settings"] = tuned_settings
        metadata["tuning"] = {
            "mode": finetune_mode,
            "baseline_pos_accuracy": round(summary["baseline"]["pos_accuracy"], 6),
            "baseline_lemma_accuracy": round(summary["baseline"]["lemma_accuracy"], 6),
            "baseline_speed": round(summary["baseline"]["speed"], 2),
            "final_pos_accuracy": round(summary["best"]["pos_accuracy"], 6),
            "final_lemma_accuracy": round(summary["best"]["lemma_accuracy"], 6),
            "final_speed": round(summary["best"]["speed"], 2),
            "evaluations": summary["evaluations"],
        }
        metadata["final_metrics"] = {
            "pos_accuracy": summary["best"]["pos_accuracy"],
            "lemma_accuracy": summary["best"]["lemma_accuracy"],
            "speed": summary["best"]["speed"],
        }
        tuning_summary = summary
        _write_model(output_path, payload)
    else:
        _write_model(output_path, payload)

    if verbose:
        stats = metadata["vocab_stats"]
        label_width = 24
        value_width = 14
        # Account for the space after │ in each column: add 1 to each width for borders
        border_top = f"┌{'─' * (label_width + 1)}┬{'─' * (value_width + 1)}┐"
        border_mid = f"├{'─' * (label_width + 1)}┼{'─' * (value_width + 1)}┤"
        border_bot = f"└{'─' * (label_width + 1)}┴{'─' * (value_width + 1)}┘"
        print("[flexipipe] training summary")
        print(border_top)
        print(f"│ {'Corpus':<{label_width}}│ {metadata['corpus_name']:<{value_width}}│")
        print(border_mid)
        print(f"│ {'Sentences':<{label_width}}│ {stats['total_sentences']:<{value_width}}│")
        print(f"│ {'Tokens':<{label_width}}│ {stats['total_tokens']:<{value_width}}│")
        print(f"│ {'Word entries':<{label_width}}│ {stats['word_entries']:<{value_width}}│")
        print(f"│ {'Multiword entries':<{label_width}}│ {stats['multiword_entries']:<{value_width}}│")
        print(f"│ {'Total entries':<{label_width}}│ {stats['total_entries']:<{value_width}}│")
        print(border_mid)
        auto_label = " (auto-selected)" if auto_select else ""
        print(f"│ {'Tag attribute':<{label_width}}│ {tag_attribute + auto_label:<{value_width}}│")
        print(border_bot)
        if len(tag_selection_entries) > 1:
            print("[flexipipe] tag attribute candidates:")
            for entry in tag_selection_entries:
                marker = "✓" if entry["tag_attribute"] == tag_attribute else " "
                pos_text = (
                    f"{entry['pos_accuracy'] * 100:.2f}%" if entry["pos_accuracy"] is not None else "   n/a "
                )
                lemma_text = (
                    f"{entry['lemma_accuracy'] * 100:.2f}%" if entry["lemma_accuracy"] is not None else "   n/a "
                )
                speed_text = (
                    f"{entry['speed']:.0f} tok/s" if entry["speed"] is not None else "n/a"
                )
                print(
                    f"  {marker} {entry['tag_attribute']}: pos={pos_text} lemma={lemma_text} speed={speed_text}"
                )
                if entry["error"]:
                    print(f"      error: {entry['error']}")
        if tuning_summary:
            print(f"[flexipipe] finetune mode: {finetune_mode} (evaluated on {evaluation_set_name} set)")
            print(
                f"  POS accuracy: {tuning_summary['baseline']['pos_accuracy']*100:.2f}% → "
                f"{tuning_summary['best']['pos_accuracy']*100:.2f}%"
            )
            print(
                f"  Lemma accuracy: {tuning_summary['baseline']['lemma_accuracy']*100:.2f}% → "
                f"{tuning_summary['best']['lemma_accuracy']*100:.2f}%"
            )
            print(
                f"  speed: {tuning_summary['baseline']['speed']:.1f} tok/s → "
                f"{tuning_summary['best']['speed']:.1f} tok/s"
            )
            print(f"  evaluations run: {tuning_summary['evaluations']}")
            print(f"  Note: Results are on {evaluation_set_name} set; test set performance may differ")

    # Clean up temporary TEITOK preparation directory if it was created (unless ud_folder was specified)
    if teitok_temp_dir and not ud_folder:
        import shutil
        try:
            shutil.rmtree(teitok_temp_dir, ignore_errors=True)
        except Exception:
            pass

    return output_path

