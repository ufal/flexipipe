#!/usr/bin/env python3
"""
Detailed accuracy analysis for flexitag models.

Provides comprehensive error analysis including:
- OOV vs in-vocab accuracy
- Error breakdown by prediction source
- Word length and frequency analysis
- Position-based analysis
- Ending-based prediction accuracy
- Beginning-based prediction accuracy (for front-inflecting languages)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tabulate import tabulate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flexipipe import Document, FlexitagFallback
from flexipipe.conllu import conllu_to_document, document_to_conllu
from flexipipe.doc import Sentence, Token


def load_vocab(model_path: Path) -> Dict:
    """Load vocabulary from model file."""
    with open(model_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return data.get("vocab", {})


def is_oov(form: str, vocab: Dict) -> bool:
    """Check if a word form is out-of-vocabulary."""
    form_lower = form.lower()
    return form not in vocab and form_lower not in vocab


def get_word_frequency(form: str, vocab: Dict) -> int:
    """Get total frequency of a word form in vocabulary."""
    total = 0
    if form in vocab:
        entry = vocab[form]
        if isinstance(entry, list):
            for item in entry:
                total += item.get("count", 0)
        else:
            total += entry.get("count", 0)
    form_lower = form.lower()
    if form_lower in vocab and form_lower != form:
        entry = vocab[form_lower]
        if isinstance(entry, list):
            for item in entry:
                total += item.get("count", 0)
        else:
            total += entry.get("count", 0)
    return total


def parse_feats(feats_str: str) -> Dict[str, str]:
    """Parse FEATS string into dictionary."""
    if not feats_str or feats_str == "_":
        return {}
    result = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key] = value
    return result


def feats_match(gold_feats: str, pred_feats: str, partial: bool = False) -> bool:
    """Check if FEATS match (exact or partial)."""
    if not gold_feats or gold_feats == "_":
        return True  # No gold feats to match
    if not pred_feats or pred_feats == "_":
        return False
    
    gold_dict = parse_feats(gold_feats)
    pred_dict = parse_feats(pred_feats)
    
    if partial:
        # Partial match: all gold features must be present in pred
        return all(pred_dict.get(k) == v for k, v in gold_dict.items())
    else:
        # Exact match
        return gold_dict == pred_dict


def get_word_beginning(form: str, length: int = 3) -> str:
    """Get word beginning (for front-inflecting languages)."""
    if len(form) <= length:
        return form
    return form[:length]


def get_word_ending(form: str, length: int = 3) -> str:
    """Get word ending (for back-inflecting languages)."""
    if len(form) <= length:
        return form
    return form[-length:]


def get_tagpos_from_model(model_path: Path) -> str:
    """Get tagpos (xpos/upos/utot) from model metadata."""
    try:
        with open(model_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        metadata = data.get("metadata", {})
        tag_attribute = metadata.get("tag_attribute", "xpos")
        # Map tag_attribute to tagpos
        if tag_attribute == "upos":
            return "upos"
        elif tag_attribute == "utot" or tag_attribute == "upos#feats":
            return "utot"
        else:
            return "xpos"
    except Exception:
        return "xpos"  # Default


def analyze_errors(
    gold_doc: Document,
    pred_doc: Document,
    vocab: Dict,
    tagpos: str = "xpos",
    verbose: bool = False,
) -> Dict:
    """Perform detailed error analysis."""
    
    # Compare at orthographic token level (not grammatical/subtoken level)
    # Orthographic tokens are the top-level tokens in sentences
    # They may have subtokens (MWTs), but we compare the orthographic token itself
    gold_ortho_tokens = []
    pred_ortho_tokens = []
    
    for sent in gold_doc.sentences:
        gold_ortho_tokens.extend(sent.tokens)
    
    for sent in pred_doc.sentences:
        pred_ortho_tokens.extend(sent.tokens)
    
    if len(gold_ortho_tokens) != len(pred_ortho_tokens):
        if verbose:
            print(f"WARNING: Orthographic token count mismatch: gold={len(gold_ortho_tokens)}, pred={len(pred_ortho_tokens)}")
        min_len = min(len(gold_ortho_tokens), len(pred_ortho_tokens))
        gold_ortho_tokens = gold_ortho_tokens[:min_len]
        pred_ortho_tokens = pred_ortho_tokens[:min_len]
    
    # Statistics
    # Note: total will be set to the number of tokens processed (where forms match)
    # This ensures we only count tokens that can be meaningfully compared
    stats = {
        "total": 0,  # Will be set to processed_count after loop
        "skipped_forms_mismatch": 0,  # Track how many tokens were skipped due to form mismatch
        "oov_total": 0,
        "oov_correct": 0,
        "in_vocab_total": 0,
        "in_vocab_correct": 0,
        "by_source": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_length": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_frequency": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_position": defaultdict(lambda: {"total": 0, "correct": 0}),
        "ending_correct": 0,
        "ending_total": 0,
        "beginning_correct": 0,
        "beginning_total": 0,
        "upos_errors": [],
        "xpos_errors": [],
        "feats_errors": [],
        "lemma_errors": [],
        "errors": [],
        "field_stats": {
            "upos": {"correct": 0, "total": 0},
            "xpos": {"correct": 0, "total": 0},
            "feats": {"correct": 0, "total": 0},
            "lemma": {"correct": 0, "total": 0},
        },
        # Per-field breakdowns for OOV and source
        "oov_by_field": {
            "upos": {"correct": 0, "total": 0},
            "xpos": {"correct": 0, "total": 0},
            "feats": {"correct": 0, "total": 0},
        },
        "in_vocab_by_field": {
            "upos": {"correct": 0, "total": 0},
            "xpos": {"correct": 0, "total": 0},
            "feats": {"correct": 0, "total": 0},
        },
        "by_source_by_field": defaultdict(lambda: {
            "upos": {"correct": 0, "total": 0},
            "xpos": {"correct": 0, "total": 0},
            "feats": {"correct": 0, "total": 0},
        }),
        # Cross-tabulation: OOV/In-Vocab by source
        "source_by_oov": defaultdict(lambda: {"oov": 0, "in_vocab": 0}),
        # Contraction statistics
        "contractions": {
            "gold_total": 0,  # Total contractions in gold
            "pred_total": 0,  # Total contractions in predicted
            "correctly_detected": 0,  # Contractions correctly detected (both are contractions)
            "parts_stats": {
                "total_parts": 0,
                "correct_parts": 0,
                "oov_parts_total": 0,
                "oov_parts_correct": 0,
                "in_vocab_parts_total": 0,
                "in_vocab_parts_correct": 0,
                "parts_by_field": {
                    "upos": {"correct": 0, "total": 0, "oov_correct": 0, "oov_total": 0, "in_vocab_correct": 0, "in_vocab_total": 0},
                    "xpos": {"correct": 0, "total": 0, "oov_correct": 0, "oov_total": 0, "in_vocab_correct": 0, "in_vocab_total": 0},
                    "feats": {"correct": 0, "total": 0, "oov_correct": 0, "oov_total": 0, "in_vocab_correct": 0, "in_vocab_total": 0},
                    "lemma": {"correct": 0, "total": 0, "oov_correct": 0, "oov_total": 0, "in_vocab_correct": 0, "in_vocab_total": 0},
                },
            },
        },
    }
    
    # Helper to get tag value based on tagpos
    def get_tag_value(token: Token, tagpos: str) -> str:
        if tagpos == "upos":
            return token.upos or ""
        elif tagpos == "utot":
            upos = token.upos or ""
            feats = token.feats or ""
            if upos and feats:
                return f"{upos}#{feats}"
            elif upos:
                return upos
            return ""
        else:  # xpos
            return token.xpos or ""
    
    # Track unknown words for debugging
    unknown_debug_count = 0
    
    # Count tokens that will actually be processed (forms match)
    processed_count = 0
    
    for i, (gold, pred) in enumerate(zip(gold_ortho_tokens, pred_ortho_tokens)):
        if gold.form != pred.form:
            stats["skipped_forms_mismatch"] += 1
            continue  # Skip if forms don't match (tokenization mismatch)
        
        processed_count += 1
        
        # Track contractions (MWTs)
        gold_is_contraction = bool(gold.subtokens and len(gold.subtokens) > 0)
        pred_is_contraction = bool(pred.subtokens and len(pred.subtokens) > 0)
        
        if gold_is_contraction:
            stats["contractions"]["gold_total"] += 1
        if pred_is_contraction:
            stats["contractions"]["pred_total"] += 1
        
        # If both are contractions, analyze the parts
        if gold_is_contraction and pred_is_contraction:
            stats["contractions"]["correctly_detected"] += 1
            
            # Analyze parts (subtokens) for correctly detected contractions
            gold_parts = gold.subtokens or []
            pred_parts = pred.subtokens or []
            
            # Align parts by position (simple alignment - assumes same number of parts)
            # For more sophisticated alignment, we could use edit distance, but for now
            # we'll compare part-by-part if they have the same count
            if len(gold_parts) == len(pred_parts):
                for gold_part, pred_part in zip(gold_parts, pred_parts):
                    # Only compare parts if their forms match (they should for correctly detected contractions)
                    if gold_part.form != pred_part.form:
                        # Skip this part - forms don't match, so it's not the same part
                        continue
                    
                    stats["contractions"]["parts_stats"]["total_parts"] += 1
                    
                    # Check OOV status for the part form
                    part_oov = is_oov(gold_part.form, vocab)
                    if part_oov:
                        stats["contractions"]["parts_stats"]["oov_parts_total"] += 1
                    else:
                        stats["contractions"]["parts_stats"]["in_vocab_parts_total"] += 1
                    
                    # Check accuracy for each field on the part
                    part_correct = True
                    
                    # UPOS
                    if gold_part.upos and gold_part.upos != "_":
                        stats["contractions"]["parts_stats"]["parts_by_field"]["upos"]["total"] += 1
                        if part_oov:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["upos"]["oov_total"] += 1
                        else:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["upos"]["in_vocab_total"] += 1
                        
                        pred_upos = pred_part.upos or ""
                        if gold_part.upos == pred_upos:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["upos"]["correct"] += 1
                            if part_oov:
                                stats["contractions"]["parts_stats"]["parts_by_field"]["upos"]["oov_correct"] += 1
                            else:
                                stats["contractions"]["parts_stats"]["parts_by_field"]["upos"]["in_vocab_correct"] += 1
                        else:
                            part_correct = False
                    
                    # XPOS - only mark incorrect if gold has a value and it doesn't match
                    stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["total"] += 1
                    if part_oov:
                        stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["oov_total"] += 1
                    else:
                        stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["in_vocab_total"] += 1
                    
                    gold_xpos = gold_part.xpos or ""
                    pred_xpos = pred_part.xpos or ""
                    if not gold_xpos or gold_xpos == "_":
                        gold_xpos = ""
                    if not pred_xpos or pred_xpos == "_":
                        pred_xpos = ""
                    
                    # Only check XPOS if gold has a value (similar to UPOS logic)
                    gold_has_xpos = bool(gold_xpos)
                    if gold_has_xpos:
                        if gold_xpos == pred_xpos:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["correct"] += 1
                            if part_oov:
                                stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["oov_correct"] += 1
                            else:
                                stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["in_vocab_correct"] += 1
                        else:
                            part_correct = False
                    else:
                        # Gold has no XPOS - consider it correct if pred also has no XPOS
                        if not pred_xpos:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["correct"] += 1
                            if part_oov:
                                stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["oov_correct"] += 1
                            else:
                                stats["contractions"]["parts_stats"]["parts_by_field"]["xpos"]["in_vocab_correct"] += 1
                        # If pred has XPOS but gold doesn't, don't mark part_correct as False
                        # (this is a prediction, not necessarily wrong)
                    
                    # FEATS - only mark incorrect if gold has a value and it doesn't match
                    stats["contractions"]["parts_stats"]["parts_by_field"]["feats"]["total"] += 1
                    if part_oov:
                        stats["contractions"]["parts_stats"]["parts_by_field"]["feats"]["oov_total"] += 1
                    else:
                        stats["contractions"]["parts_stats"]["parts_by_field"]["feats"]["in_vocab_total"] += 1
                    
                    gold_feats = gold_part.feats or ""
                    pred_feats = pred_part.feats or ""
                    if not gold_feats or gold_feats == "_":
                        gold_feats = ""
                    if not pred_feats or pred_feats == "_":
                        pred_feats = ""
                    
                    # FEATS comparison: empty feats is valid, so compare directly
                    if gold_feats == pred_feats:
                        stats["contractions"]["parts_stats"]["parts_by_field"]["feats"]["correct"] += 1
                        if part_oov:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["feats"]["oov_correct"] += 1
                        else:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["feats"]["in_vocab_correct"] += 1
                    else:
                        # Only mark incorrect if gold has feats (empty feats is valid)
                        if gold_feats:
                            part_correct = False
                    
                    # Lemma
                    if gold_part.lemma and gold_part.lemma != "_":
                        stats["contractions"]["parts_stats"]["parts_by_field"]["lemma"]["total"] += 1
                        if part_oov:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["lemma"]["oov_total"] += 1
                        else:
                            stats["contractions"]["parts_stats"]["parts_by_field"]["lemma"]["in_vocab_total"] += 1
                        
                        pred_lemma = pred_part.lemma or ""
                        if not pred_lemma or pred_lemma == "_":
                            pred_lemma = ""
                        gold_lemma = gold_part.lemma if gold_part.lemma != "_" else ""
                        if pred_lemma.lower() == gold_lemma.lower():
                            stats["contractions"]["parts_stats"]["parts_by_field"]["lemma"]["correct"] += 1
                            if part_oov:
                                stats["contractions"]["parts_stats"]["parts_by_field"]["lemma"]["oov_correct"] += 1
                            else:
                                stats["contractions"]["parts_stats"]["parts_by_field"]["lemma"]["in_vocab_correct"] += 1
                        else:
                            part_correct = False
                    
                    # Overall part correctness (all fields match)
                    if part_correct:
                        stats["contractions"]["parts_stats"]["correct_parts"] += 1
                        if part_oov:
                            stats["contractions"]["parts_stats"]["oov_parts_correct"] += 1
                        else:
                            stats["contractions"]["parts_stats"]["in_vocab_parts_correct"] += 1
        
        # Check OOV status
        oov = is_oov(gold.form, vocab)
        
        # Debug output for unknown words (first 10) - only in verbose mode
        source = pred.source or "unknown"
        if verbose and source == "unknown" and unknown_debug_count < 10:
            unknown_debug_count += 1
            gold_tag = get_tag_value(gold, tagpos)
            pred_tag = get_tag_value(pred, tagpos)
            print(f"[debug] Unknown word #{unknown_debug_count}: '{gold.form}'")
            print(f"  Gold: {tagpos}={gold_tag}, upos={gold.upos}, xpos={gold.xpos}, feats={gold.feats}, lemma={gold.lemma}")
            print(f"  Pred: {tagpos}={pred_tag}, upos={pred.upos}, xpos={pred.xpos}, feats={pred.feats}, lemma={pred.lemma}")
            print(f"  Source: {source}")
            # Check if input token had tags (data leakage check)
            # We can't check the input token directly here, but we can check if pred matches gold exactly
            exact_match = (gold_tag == pred_tag and 
                          (gold.upos or "") == (pred.upos or "") and
                          (gold.xpos or "") == (pred.xpos or "") and
                          (gold.feats or "") == (pred.feats or ""))
            print(f"  Match: {tagpos}={('✓' if gold_tag == pred_tag else '✗')}, upos={('✓' if gold.upos == (pred.upos or '') else '✗')}, xpos={('✓' if gold.xpos == (pred.xpos or '') else '✗')}")
            if exact_match:
                print(f"  WARNING: Exact match - possible data leakage!")
            print()
        if oov:
            stats["oov_total"] += 1
        else:
            stats["in_vocab_total"] += 1
        
        # Get frequency bucket
        freq = get_word_frequency(gold.form, vocab)
        if freq == 0:
            freq_bucket = "0"
        elif freq < 5:
            freq_bucket = "1-4"
        elif freq < 20:
            freq_bucket = "5-19"
        elif freq < 100:
            freq_bucket = "20-99"
        else:
            freq_bucket = "100+"
        
        # Get length bucket
        length = len(gold.form)
        if length <= 3:
            length_bucket = "1-3"
        elif length <= 6:
            length_bucket = "4-6"
        elif length <= 10:
            length_bucket = "7-10"
        else:
            length_bucket = "11+"
        
        # Position in sentence (approximate)
        sent_idx = i // 20  # Rough sentence position
        if sent_idx == 0:
            pos_bucket = "start"
        elif sent_idx < 3:
            pos_bucket = "early"
        elif sent_idx < 7:
            pos_bucket = "middle"
        else:
            pos_bucket = "late"
        
        # Check accuracy for each field (matching check.py logic)
        # Only count if gold has the field
        # UPOS: only count if gold has upos (matching check.py line 200-201)
        if gold.upos and gold.upos != "_":
            stats["field_stats"]["upos"]["total"] += 1
            pred_upos = pred.upos or ""
            if gold.upos == pred_upos:
                stats["field_stats"]["upos"]["correct"] += 1
            else:
                # Track error
                stats["upos_errors"].append({
                    "form": gold.form,
                    "gold": gold.upos,
                    "pred": pred_upos,
                    "source": source,
                    "oov": oov,
                })
        
        # XPOS: count ALL tokens (XPOS should be present for all)
        stats["field_stats"]["xpos"]["total"] += 1
        pred_xpos = pred.xpos or ""
        gold_xpos = gold.xpos or ""
        # Normalize empty string and "_" to empty for comparison
        if not pred_xpos or pred_xpos == "_":
            pred_xpos = ""
        if not gold_xpos or gold_xpos == "_":
            gold_xpos = ""
        if gold_xpos == pred_xpos:
            stats["field_stats"]["xpos"]["correct"] += 1
        else:
            # Track error
            stats["xpos_errors"].append({
                "form": gold.form,
                "gold": gold_xpos if gold_xpos else "(empty)",
                "pred": pred_xpos if pred_xpos else "(empty)",
                "source": source,
                "oov": oov,
            })
        
        # FEATS: count ALL tokens (empty feats is a valid value)
        # If gold has empty feats and prediction is empty → correct
        # If gold has empty feats and prediction has something → wrong
        # If gold has feats and prediction is empty → wrong
        # If gold has feats and prediction matches → correct
        stats["field_stats"]["feats"]["total"] += 1
        pred_feats = pred.feats or ""
        gold_feats = gold.feats or ""
        # Normalize empty string and "_" to empty for comparison
        if not pred_feats or pred_feats == "_":
            pred_feats = ""
        if not gold_feats or gold_feats == "_":
            gold_feats = ""
        # Compare normalized values
        if gold_feats == pred_feats:
            stats["field_stats"]["feats"]["correct"] += 1
        else:
            # Track error - always track mismatches
            stats["feats_errors"].append({
                "form": gold.form,
                "gold": gold_feats if gold_feats else "(empty)",
                "pred": pred_feats if pred_feats else "(empty)",
                "source": source,
                "oov": oov,
            })
        
        # Lemma: only count if gold has lemma (matching check.py line 206-207)
        # CRITICAL: Only count if gold has lemma AND it's not empty/"_"
        if gold.lemma and gold.lemma != "_":
            stats["field_stats"]["lemma"]["total"] += 1
            # check.py: (pred_tok.lemma or "") == gold_tok.lemma
            pred_lemma = pred.lemma or ""
            # Normalize empty string and "_" to empty for comparison
            if not pred_lemma or pred_lemma == "_":
                pred_lemma = ""
            gold_lemma_normalized = gold.lemma if gold.lemma != "_" else ""
            # CRITICAL: Compare normalized values (case-insensitive for lemma)
            if pred_lemma.lower() == gold_lemma_normalized.lower():
                stats["field_stats"]["lemma"]["correct"] += 1
            else:
                # Track error - always track mismatches
                stats["lemma_errors"].append({
                    "form": gold.form,
                    "gold": gold.lemma,
                    "pred": pred_lemma if pred_lemma else "(empty)",
                    "source": source,
                    "oov": oov,
                })
        
        # For overall correctness check (used for breakdowns)
        upos_correct = (gold.upos == pred.upos or not gold.upos or gold.upos == "_")
        xpos_correct = (gold.xpos == pred.xpos or not gold.xpos or gold.xpos == "_")
        feats_correct = feats_match(gold.feats, pred.feats, partial=True)
        lemma_correct = (gold.lemma.lower() == pred.lemma.lower() or not gold.lemma or gold.lemma == "_")
        
        # Overall correct (all fields match) - stricter definition
        overall_correct = upos_correct and xpos_correct and feats_correct and lemma_correct
        
        # Track by source (overall accuracy)
        # For orthographic tokens, use the token's source directly
        # (not subtoken sources, which are only for grammatical tokens)
        source = pred.source or "unknown"
        stats["by_source"][source]["total"] += 1
        if overall_correct:
            stats["by_source"][source]["correct"] += 1
        
        # Track OOV/In-Vocab breakdown by source
        if oov:
            stats["source_by_oov"][source]["oov"] += 1
            if overall_correct:
                stats["oov_correct"] += 1
        else:
            stats["source_by_oov"][source]["in_vocab"] += 1
            if overall_correct:
                stats["in_vocab_correct"] += 1
        
        # Track OOV contractions for debugging
        if source.startswith("contractions:") and oov:
            if "oov_contractions" not in stats:
                stats["oov_contractions"] = []
            
            # For contractions, tags may be on subtokens, not the parent token
            # Extract tags from subtokens if available
            pred_xpos = pred.xpos or ""
            pred_upos = pred.upos or ""
            
            # If parent token has no tags but has subtokens, extract from subtokens
            if (not pred_xpos or not pred_upos) and pred.subtokens:
                # Build combined tags from subtokens
                subtoken_xpos = []
                subtoken_upos = []
                for st in pred.subtokens:
                    if st.xpos and st.xpos != "_":
                        subtoken_xpos.append(st.xpos)
                    if st.upos and st.upos != "_":
                        subtoken_upos.append(st.upos)
                if subtoken_xpos:
                    pred_xpos = ".".join(subtoken_xpos) if len(subtoken_xpos) > 1 else subtoken_xpos[0]
                if subtoken_upos:
                    pred_upos = ".".join(subtoken_upos) if len(subtoken_upos) > 1 else subtoken_upos[0]
            
            # Also extract gold tags from subtokens if needed
            gold_xpos = gold.xpos or ""
            gold_upos = gold.upos or ""
            if (not gold_xpos or not gold_upos) and gold.subtokens:
                subtoken_xpos = []
                subtoken_upos = []
                for st in gold.subtokens:
                    if st.xpos and st.xpos != "_":
                        subtoken_xpos.append(st.xpos)
                    if st.upos and st.upos != "_":
                        subtoken_upos.append(st.upos)
                if subtoken_xpos:
                    gold_xpos = ".".join(subtoken_xpos) if len(subtoken_xpos) > 1 else subtoken_xpos[0]
                if subtoken_upos:
                    gold_upos = ".".join(subtoken_upos) if len(subtoken_upos) > 1 else subtoken_upos[0]
            
            stats["oov_contractions"].append({
                "form": pred.form,
                "source": source,
                "gold_xpos": gold_xpos,
                "pred_xpos": pred_xpos,
                "gold_upos": gold_upos,
                "pred_upos": pred_upos,
            })
        
        # Track per-field accuracies for OOV/in-vocab
        if oov:
            if gold.upos and gold.upos != "_":
                stats["oov_by_field"]["upos"]["total"] += 1
                if gold.upos == (pred.upos or ""):
                    stats["oov_by_field"]["upos"]["correct"] += 1
            if gold.xpos and gold.xpos != "_":
                stats["oov_by_field"]["xpos"]["total"] += 1
                if gold.xpos == (pred.xpos or ""):
                    stats["oov_by_field"]["xpos"]["correct"] += 1
            if gold.feats and gold.feats != "_":
                stats["oov_by_field"]["feats"]["total"] += 1
                pred_feats = pred.feats or ""
                if not pred_feats or pred_feats == "_":
                    pred_feats = ""
                if gold.feats == pred_feats:
                    stats["oov_by_field"]["feats"]["correct"] += 1
        else:
            if gold.upos and gold.upos != "_":
                stats["in_vocab_by_field"]["upos"]["total"] += 1
                if gold.upos == (pred.upos or ""):
                    stats["in_vocab_by_field"]["upos"]["correct"] += 1
            if gold.xpos and gold.xpos != "_":
                stats["in_vocab_by_field"]["xpos"]["total"] += 1
                if gold.xpos == (pred.xpos or ""):
                    stats["in_vocab_by_field"]["xpos"]["correct"] += 1
            if gold.feats and gold.feats != "_":
                stats["in_vocab_by_field"]["feats"]["total"] += 1
                pred_feats = pred.feats or ""
                if not pred_feats or pred_feats == "_":
                    pred_feats = ""
                if gold.feats == pred_feats:
                    stats["in_vocab_by_field"]["feats"]["correct"] += 1
        
        # Track per-field accuracies for each source
        if gold.upos and gold.upos != "_":
            stats["by_source_by_field"][source]["upos"]["total"] += 1
            if gold.upos == (pred.upos or ""):
                stats["by_source_by_field"][source]["upos"]["correct"] += 1
        if gold.xpos and gold.xpos != "_":
            stats["by_source_by_field"][source]["xpos"]["total"] += 1
            if gold.xpos == (pred.xpos or ""):
                stats["by_source_by_field"][source]["xpos"]["correct"] += 1
        if gold.feats and gold.feats != "_":
            stats["by_source_by_field"][source]["feats"]["total"] += 1
            pred_feats = pred.feats or ""
            if not pred_feats or pred_feats == "_":
                pred_feats = ""
            if gold.feats == pred_feats:
                stats["by_source_by_field"][source]["feats"]["correct"] += 1
        
        # Track by length (using tagpos-based accuracy)
        # CRITICAL: Only count if gold has the tagpos field
        gold_tag = get_tag_value(gold, tagpos)
        pred_tag = get_tag_value(pred, tagpos)
        # Only evaluate tagpos accuracy if gold has a value for it
        if gold_tag and gold_tag != "_":
            tag_correct = (gold_tag == pred_tag)
            stats["by_length"][length_bucket]["total"] += 1
            if tag_correct:
                stats["by_length"][length_bucket]["correct"] += 1
            
            # Track by frequency (using tagpos-based accuracy)
            stats["by_frequency"][freq_bucket]["total"] += 1
            if tag_correct:
                stats["by_frequency"][freq_bucket]["correct"] += 1
            
            # Track by position (using tagpos-based accuracy)
            stats["by_position"][pos_bucket]["total"] += 1
            if tag_correct:
                stats["by_position"][pos_bucket]["correct"] += 1
            
            # Check ending-based predictions (using tagpos-based accuracy)
            # Look for sources that indicate ending-based heuristics
            if "ending" in source.lower() or source == "endings" or "end" in source.lower():
                stats["ending_total"] += 1
                if tag_correct:
                    stats["ending_correct"] += 1
        
        # Check beginning-based predictions (would need to be tracked separately)
        # For now, we'll analyze if beginning would help
        if not overall_correct and oov:
            gold_beginning = get_word_beginning(gold.form, 3)
            pred_beginning = get_word_beginning(pred.form, 3)
            if gold_beginning == pred_beginning:
                stats["beginning_total"] += 1
                # Check if beginning match would have helped
                # (This is heuristic - we'd need to track actual beginning-based predictions)
        
        # Track individual errors
        if not upos_correct:
            stats["upos_errors"].append({
                "form": gold.form,
                "gold": gold.upos,
                "pred": pred.upos,
                "source": source,
                "oov": oov,
            })
        if not xpos_correct:
            stats["xpos_errors"].append({
                "form": gold.form,
                "gold": gold.xpos,
                "pred": pred.xpos,
                "source": source,
                "oov": oov,
            })
        if not feats_correct:
            stats["feats_errors"].append({
                "form": gold.form,
                "gold": gold.feats,
                "pred": pred.feats,
                "source": source,
                "oov": oov,
            })
        if not lemma_correct:
            stats["lemma_errors"].append({
                "form": gold.form,
                "gold": gold.lemma,
                "pred": pred.lemma,
                "source": source,
                "oov": oov,
            })
        
        if not overall_correct:
            stats["errors"].append({
                "form": gold.form,
                "position": i,
                "oov": oov,
                "source": source,
                "upos": (gold.upos, pred.upos),
                "xpos": (gold.xpos, pred.xpos),
                "feats": (gold.feats, pred.feats),
                "lemma": (gold.lemma, pred.lemma),
            })
    
    # Set total to the actual number of tokens processed (where forms matched)
    stats["total"] = processed_count
    
    return stats


def print_report(stats: Dict, tagpos: str = "xpos", verbose: bool = False):
    """Print detailed accuracy report."""
    
    total = stats["total"]
    if total == 0:
        print("No tokens to analyze")
        return
    
    print("\n" + "=" * 80)
    print("DETAILED ACCURACY ANALYSIS")
    print("=" * 80)
    
    # Print timing information (always show, even if elapsed_seconds is 0)
    elapsed_seconds = stats.get("elapsed_seconds", 0.0)
    word_count = stats.get("word_count", total)
    sentence_count = stats.get("sentence_count", 0)
    speed = stats.get("speed", 0.0)
    sent_speed = stats.get("sent_speed", 0.0)
    
    # If sentence_count is 0, use estimate as fallback (shouldn't happen if set correctly)
    if sentence_count == 0 and word_count > 0:
        # Estimate from token count (roughly 30 tokens per sentence for Spanish)
        sentence_count = max(1, word_count // 30)
    
    print("\nAnnotation Details")
    print(f"Tokens: {word_count:,}")
    print(f"Sentences: {sentence_count:,}")
    
    # Calculate tokens per sentence
    tok_per_sent = word_count / sentence_count if sentence_count > 0 else 0.0
    
    # Show both C++ timing and total timing if available
    total_timing = stats.get("total_elapsed_seconds", 0.0) or 0.0
    cpp_timing = stats.get("elapsed_seconds", 0.0) or 0.0
    
    # Use the timing values from stats dict (they should be set earlier)
    if cpp_timing > 0:
        print(f"Timing: {cpp_timing:.3f} seconds (C++ core)")
        if total_timing > cpp_timing * 1.1:  # Show total if significantly different
            print(f"        {total_timing:.3f} seconds (total, including Python overhead)")
        if speed > 0:
            print(f"Speed: {speed:,.0f} tok/s")
            print(f"       {sent_speed:,.2f} sent/s")
            print(f"       {tok_per_sent:.2f} tok/sent")
    elif total_timing > 0:
        # Use total timing if C++ timing not available
        print(f"Timing: {total_timing:.3f} seconds (total)")
        if speed > 0:
            print(f"Speed: {speed:,.0f} tok/s")
            print(f"       {sent_speed:,.2f} sent/s")
            print(f"       {tok_per_sent:.2f} tok/sent")
    else:
        print("Timing: (not available)")
        if sentence_count > 0:
            print(f"       {tok_per_sent:.2f} tok/sent")
    print()
    
    # Report tokenization alignment
    skipped = stats.get("skipped_forms_mismatch", 0)
    gold_token_count = stats.get("gold_token_count", total)
    if skipped > 0 or gold_token_count != total:
        print("\n" + "-" * 80)
        print("Tokenization Alignment")
        print("-" * 80)
        print(f"Total tokens in gold: {gold_token_count}")
        print(f"Tokens with matching forms: {total}")
        print(f"Tokens skipped (form mismatch): {skipped}")
        if gold_token_count > 0:
            alignment_pct = (total / gold_token_count * 100)
            print(f"Alignment: {alignment_pct:.2f}%")
            if alignment_pct < 50:
                print("WARNING: Less than 50% alignment - significant tokenization mismatch!")
                print("         This may indicate tokenization differences between gold and predicted.")
        print()
    
    # Calculate per-field accuracies (matching check/train scripts)
    field_stats = stats["field_stats"]
    upos_acc = (field_stats["upos"]["correct"] / field_stats["upos"]["total"] * 100) if field_stats["upos"]["total"] > 0 else 0
    xpos_acc = (field_stats["xpos"]["correct"] / field_stats["xpos"]["total"] * 100) if field_stats["xpos"]["total"] > 0 else 0
    feats_acc = (field_stats["feats"]["correct"] / field_stats["feats"]["total"] * 100) if field_stats["feats"]["total"] > 0 else 0
    lemma_acc = (field_stats["lemma"]["correct"] / field_stats["lemma"]["total"] * 100) if field_stats["lemma"]["total"] > 0 else 0
    
    # Overall accuracy (all fields must match - stricter)
    overall_correct = stats["oov_correct"] + stats["in_vocab_correct"]
    overall_acc = (overall_correct / total * 100) if total > 0 else 0
    
    print("\n" + "-" * 80)
    print("Accuracy by Field (matching check/train scripts)")
    print("-" * 80)
    print(f"UPOS:  {field_stats['upos']['correct']}/{field_stats['upos']['total']} ({upos_acc:.2f}%)")
    print(f"XPOS:  {field_stats['xpos']['correct']}/{field_stats['xpos']['total']} ({xpos_acc:.2f}%)")
    print(f"FEATS: {field_stats['feats']['correct']}/{field_stats['feats']['total']} ({feats_acc:.2f}%)")
    print(f"Lemma: {field_stats['lemma']['correct']}/{field_stats['lemma']['total']} ({lemma_acc:.2f}%)")
    print(f"\nOverall (all fields must match): {overall_correct}/{total} ({overall_acc:.2f}%)")
    print(f"\nNOTE: Using {tagpos.upper()} for breakdown calculations (from model metadata).")
    print("      The 'Overall' accuracy requires ALL fields (UPOS, XPOS, FEATS, Lemma) to match.")
    print("      Individual field accuracies match what check/train scripts report.")
    print(f"      Breakdowns (by length, frequency, position, ending) use {tagpos.upper()} accuracy only.")
    print(f"      OOV and source breakdowns use 'all fields' accuracy.")
    
    # OOV vs In-Vocab
    print("\n" + "-" * 80)
    print("OOV vs In-Vocabulary")
    print("-" * 80)
    oov_acc = (stats["oov_correct"] / stats["oov_total"] * 100) if stats["oov_total"] > 0 else 0
    invocab_acc = (stats["in_vocab_correct"] / stats["in_vocab_total"] * 100) if stats["in_vocab_total"] > 0 else 0
    print(f"OOV (overall):           {stats['oov_correct']}/{stats['oov_total']} ({oov_acc:.2f}%)")
    print(f"In-Vocabulary (overall): {stats['in_vocab_correct']}/{stats['in_vocab_total']} ({invocab_acc:.2f}%)")
    
    # Per-field accuracies for OOV
    oov_by_field = stats.get("oov_by_field", {})
    if oov_by_field.get("upos", {}).get("total", 0) > 0:
        oov_upos_acc = (oov_by_field["upos"]["correct"] / oov_by_field["upos"]["total"] * 100)
        print(f"  OOV UPOS:  {oov_by_field['upos']['correct']}/{oov_by_field['upos']['total']} ({oov_upos_acc:.2f}%)")
    if oov_by_field.get("xpos", {}).get("total", 0) > 0:
        oov_xpos_acc = (oov_by_field["xpos"]["correct"] / oov_by_field["xpos"]["total"] * 100)
        print(f"  OOV XPOS:  {oov_by_field['xpos']['correct']}/{oov_by_field['xpos']['total']} ({oov_xpos_acc:.2f}%)")
    if oov_by_field.get("feats", {}).get("total", 0) > 0:
        oov_feats_acc = (oov_by_field["feats"]["correct"] / oov_by_field["feats"]["total"] * 100)
        print(f"  OOV FEATS: {oov_by_field['feats']['correct']}/{oov_by_field['feats']['total']} ({oov_feats_acc:.2f}%)")
    
    # Per-field accuracies for In-Vocab
    invocab_by_field = stats.get("in_vocab_by_field", {})
    if invocab_by_field.get("upos", {}).get("total", 0) > 0:
        invocab_upos_acc = (invocab_by_field["upos"]["correct"] / invocab_by_field["upos"]["total"] * 100)
        print(f"  In-Vocab UPOS:  {invocab_by_field['upos']['correct']}/{invocab_by_field['upos']['total']} ({invocab_upos_acc:.2f}%)")
    if invocab_by_field.get("xpos", {}).get("total", 0) > 0:
        invocab_xpos_acc = (invocab_by_field["xpos"]["correct"] / invocab_by_field["xpos"]["total"] * 100)
        print(f"  In-Vocab XPOS:  {invocab_by_field['xpos']['correct']}/{invocab_by_field['xpos']['total']} ({invocab_xpos_acc:.2f}%)")
    if invocab_by_field.get("feats", {}).get("total", 0) > 0:
        invocab_feats_acc = (invocab_by_field["feats"]["correct"] / invocab_by_field["feats"]["total"] * 100)
        print(f"  In-Vocab FEATS: {invocab_by_field['feats']['correct']}/{invocab_by_field['feats']['total']} ({invocab_feats_acc:.2f}%)")
    
    # Contractions section (only if there are contractions)
    contractions = stats.get("contractions", {})
    gold_contractions = contractions.get("gold_total", 0)
    pred_contractions = contractions.get("pred_total", 0)
    correctly_detected = contractions.get("correctly_detected", 0)
    
    if gold_contractions > 0 or pred_contractions > 0:
        print("\n" + "-" * 80)
        print("Contractions (MWTs)")
        print("-" * 80)
        
        # Recall and Precision
        recall = (correctly_detected / gold_contractions * 100) if gold_contractions > 0 else 0.0
        precision = (correctly_detected / pred_contractions * 100) if pred_contractions > 0 else 0.0
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
        
        print(f"Gold contractions:      {gold_contractions}")
        print(f"Predicted contractions: {pred_contractions}")
        print(f"Correctly detected:     {correctly_detected}")
        print(f"Recall:                 {recall:.2f}%")
        print(f"Precision:              {precision:.2f}%")
        print(f"F1:                     {f1:.2f}%")
        
        # Parts accuracy (only for correctly detected contractions)
        parts_stats = contractions.get("parts_stats", {})
        total_parts = parts_stats.get("total_parts", 0)
        correct_parts = parts_stats.get("correct_parts", 0)
        oov_parts_total = parts_stats.get("oov_parts_total", 0)
        oov_parts_correct = parts_stats.get("oov_parts_correct", 0)
        in_vocab_parts_total = parts_stats.get("in_vocab_parts_total", 0)
        in_vocab_parts_correct = parts_stats.get("in_vocab_parts_correct", 0)
        
        if total_parts > 0:
            print(f"\nParts accuracy (within correctly detected contractions):")
            parts_acc = (correct_parts / total_parts * 100) if total_parts > 0 else 0.0
            print(f"  Overall:              {correct_parts}/{total_parts} ({parts_acc:.2f}%)")
            
            if oov_parts_total > 0:
                oov_parts_acc = (oov_parts_correct / oov_parts_total * 100)
                print(f"  OOV parts:            {oov_parts_correct}/{oov_parts_total} ({oov_parts_acc:.2f}%)")
            
            if in_vocab_parts_total > 0:
                in_vocab_parts_acc = (in_vocab_parts_correct / in_vocab_parts_total * 100)
                print(f"  In-vocab parts:       {in_vocab_parts_correct}/{in_vocab_parts_total} ({in_vocab_parts_acc:.2f}%)")
            
            # Per-field accuracies for parts
            parts_by_field = parts_stats.get("parts_by_field", {})
            print(f"\n  Parts accuracy by field:")
            
            for field_name in ["upos", "xpos", "feats", "lemma"]:
                field_stats = parts_by_field.get(field_name, {})
                field_total = field_stats.get("total", 0)
                if field_total > 0:
                    field_correct = field_stats.get("correct", 0)
                    field_acc = (field_correct / field_total * 100)
                    
                    oov_total = field_stats.get("oov_total", 0)
                    oov_correct = field_stats.get("oov_correct", 0)
                    oov_acc = (oov_correct / oov_total * 100) if oov_total > 0 else 0.0
                    
                    in_vocab_total = field_stats.get("in_vocab_total", 0)
                    in_vocab_correct = field_stats.get("in_vocab_correct", 0)
                    in_vocab_acc = (in_vocab_correct / in_vocab_total * 100) if in_vocab_total > 0 else 0.0
                    
                    print(f"    {field_name.upper()}:")
                    print(f"      Overall:          {field_correct}/{field_total} ({field_acc:.2f}%)")
                    if oov_total > 0:
                        print(f"      OOV:              {oov_correct}/{oov_total} ({oov_acc:.2f}%)")
                    if in_vocab_total > 0:
                        print(f"      In-vocab:         {in_vocab_correct}/{in_vocab_total} ({in_vocab_acc:.2f}%)")
    
    # By source
    print("\n" + "-" * 80)
    print("Accuracy by Prediction Source")
    print("-" * 80)
    source_data = []
    for source, data in sorted(stats["by_source"].items()):
        acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        source_data.append([source, data["correct"], data["total"], f"{acc:.2f}%"])
    print(tabulate(source_data, headers=["Source", "Correct", "Total", "Accuracy (overall)"]))
    
    # Per-field accuracies by source
    print("\nPer-field accuracies by source:")
    by_source_by_field = stats.get("by_source_by_field", {})
    for source in sorted(by_source_by_field.keys()):
        source_fields = by_source_by_field[source]
        field_accs = []
        if source_fields.get("upos", {}).get("total", 0) > 0:
            upos_acc = (source_fields["upos"]["correct"] / source_fields["upos"]["total"] * 100)
            field_accs.append(f"UPOS: {upos_acc:.2f}%")
        if source_fields.get("xpos", {}).get("total", 0) > 0:
            xpos_acc = (source_fields["xpos"]["correct"] / source_fields["xpos"]["total"] * 100)
            field_accs.append(f"XPOS: {xpos_acc:.2f}%")
        if source_fields.get("feats", {}).get("total", 0) > 0:
            feats_acc = (source_fields["feats"]["correct"] / source_fields["feats"]["total"] * 100)
            field_accs.append(f"FEATS: {feats_acc:.2f}%")
        if field_accs:
            print(f"  {source}: {', '.join(field_accs)}")
    
    # Cross-tabulation: OOV/In-Vocab by source
    print("\n" + "-" * 80)
    print("OOV/In-Vocab Breakdown by Source")
    print("-" * 80)
    print("(Shows how many OOV vs In-Vocab words use each prediction source)")
    print()
    source_by_oov = stats.get("source_by_oov", {})
    if source_by_oov:
        cross_tab_data = []
        for source in sorted(source_by_oov.keys()):
            oov_count = source_by_oov[source].get("oov", 0)
            in_vocab_count = source_by_oov[source].get("in_vocab", 0)
            total_count = oov_count + in_vocab_count
            if total_count > 0:
                cross_tab_data.append([
                    source,
                    oov_count,
                    in_vocab_count,
                    total_count,
                    f"{(oov_count / total_count * 100):.1f}%" if total_count > 0 else "0%"
                ])
        if cross_tab_data:
            print(tabulate(
                cross_tab_data,
                headers=["Source", "OOV", "In-Vocab", "Total", "OOV %"],
                tablefmt="simple"
            ))
            print()
            print("NOTE: OOV/In-Vocab is determined by whether the word form exists in the vocabulary.")
            print("      Source is determined by how the tagger made its prediction.")
            print("      These can differ because:")
            print("      - OOV words may use 'lexicon' source if normalized/matched to similar words")
            print("      - In-Vocab words may use 'ending' source if no good match found in lexicon")
    
    # By word length
    print("\n" + "-" * 80)
    print("Accuracy by Word Length")
    print("-" * 80)
    length_data = []
    for length_bucket in ["1-3", "4-6", "7-10", "11+"]:
        data = stats["by_length"][length_bucket]
        acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        length_data.append([length_bucket, data["correct"], data["total"], f"{acc:.2f}%"])
    print(tabulate(length_data, headers=["Length", "Correct", "Total", "Accuracy"]))
    
    # By frequency
    print("\n" + "-" * 80)
    print("Accuracy by Word Frequency")
    print("-" * 80)
    freq_data = []
    for freq_bucket in ["0", "1-4", "5-19", "20-99", "100+"]:
        data = stats["by_frequency"][freq_bucket]
        acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        freq_data.append([freq_bucket, data["correct"], data["total"], f"{acc:.2f}%"])
    print(tabulate(freq_data, headers=["Frequency", "Correct", "Total", "Accuracy"]))
    
    # By position
    print("\n" + "-" * 80)
    print("Accuracy by Position in Sentence")
    print("-" * 80)
    pos_data = []
    for pos_bucket in ["start", "early", "middle", "late"]:
        data = stats["by_position"][pos_bucket]
        acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        pos_data.append([pos_bucket, data["correct"], data["total"], f"{acc:.2f}%"])
    print(tabulate(pos_data, headers=["Position", "Correct", "Total", "Accuracy"]))
    
    # Ending-based predictions
    print("\n" + "-" * 80)
    print("Ending-based Predictions")
    print("-" * 80)
    if stats["ending_total"] > 0:
        ending_acc = (stats["ending_correct"] / stats["ending_total"] * 100)
        print(f"Accuracy: {stats['ending_correct']}/{stats['ending_total']} ({ending_acc:.2f}%)")
        print(f"Coverage: {stats['ending_total']}/{total} ({stats['ending_total']/total*100:.2f}%)")
    else:
        print("No ending-based predictions found")
        print("(This may indicate ending heuristics are not being used)")
    
    # Report OOV contractions if any
    oov_contractions = stats.get("oov_contractions", [])
    if oov_contractions:
        print("\n" + "-" * 80)
        print("OOV CONTRACTIONS (first 20 examples)")
        print("-" * 80)
        for i, ex in enumerate(oov_contractions[:20]):
            print(f"{i+1}. '{ex['form']}' - source: {ex['source']}")
            print(f"   Gold: xpos={ex['gold_xpos']}, upos={ex['gold_upos']}")
            print(f"   Pred: xpos={ex['pred_xpos']}, upos={ex['pred_upos']}")
        if len(oov_contractions) > 20:
            print(f"\n... and {len(oov_contractions) - 20} more OOV contractions")
        print(f"\nTotal OOV contractions: {len(oov_contractions)}")
        print()
    
    # Warning about data leakage
    unknown_source = stats["by_source"].get("unknown", {"total": 0, "correct": 0})
    fallback_source = stats["by_source"].get("fallback", {"total": 0, "correct": 0})
    existing_tag_source = stats["by_source"].get("existing tag", {"total": 0, "correct": 0})
    
    if (unknown_source["total"] > 0 and unknown_source["correct"] == unknown_source["total"]) or \
       (fallback_source["total"] > 0 and fallback_source["correct"] == fallback_source["total"]) or \
       (existing_tag_source["total"] > 0 and existing_tag_source["correct"] == existing_tag_source["total"]):
        print("\n" + "!" * 80)
        print("WARNING: Possible data leakage detected!")
        print("!" * 80)
        if unknown_source["total"] > 0 and unknown_source["correct"] == unknown_source["total"]:
            print(f"  'unknown' source: {unknown_source['correct']}/{unknown_source['total']} (100%)")
            print("  This is suspicious - unknown words should not be 100% accurate.")
            print("  This may indicate:")
            print("    - Tags were not properly cleared from input")
            print("    - The tagger is somehow using gold tag information")
            print("    - Very unlikely: all unknown words happened to get the right tag by chance")
        if existing_tag_source["total"] > 0 and existing_tag_source["correct"] == existing_tag_source["total"]:
            print(f"  'existing tag' source: {existing_tag_source['correct']}/{existing_tag_source['total']} (100%)")
            print("  This indicates the input had tags that were preserved.")
        if fallback_source["total"] > 0 and fallback_source["correct"] == fallback_source["total"]:
            print(f"  'fallback' source: {fallback_source['correct']}/{fallback_source['total']} (100%)")
            print("  This is suspicious - fallback should not be 100% accurate.")
        print("  Make sure the input file is detagged before tagging!")
        print("!" * 80)
    
    # Verbose: show actual errors
    if verbose:
        print("\n" + "=" * 80)
        print("DETAILED ERROR LIST")
        print("=" * 80)
        
        # Show first 50 errors
        for i, error in enumerate(stats["errors"][:50]):
            print(f"\nError {i+1}:")
            print(f"  Form: {error['form']}")
            print(f"  OOV: {error['oov']}")
            print(f"  Source: {error['source']}")
            if error['upos'][0] != error['upos'][1]:
                print(f"  UPOS: {error['upos'][0]} -> {error['upos'][1]}")
            if error['xpos'][0] != error['xpos'][1]:
                print(f"  XPOS: {error['xpos'][0]} -> {error['xpos'][1]}")
            if error['feats'][0] != error['feats'][1]:
                print(f"  FEATS: {error['feats'][0]} -> {error['feats'][1]}")
            if error['lemma'][0] != error['lemma'][1]:
                print(f"  Lemma: {error['lemma'][0]} -> {error['lemma'][1]}")
        
        if len(stats["errors"]) > 50:
            print(f"\n... and {len(stats['errors']) - 50} more errors")


def main():
    parser = argparse.ArgumentParser(
        description="Detailed accuracy analysis for flexitag models"
    )
    parser.add_argument("--model", required=True, help="Path to model file (JSON)")
    parser.add_argument("--test", required=True, help="Path to test CoNLL-U file")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--debug", action="store_true", help="Show detailed error list")
    parser.add_argument("--output", help="Path to output file (optional)")
    parser.add_argument(
        "--endlen",
        type=int,
        default=None,
        help="Override endlen parameter (0 to disable word ending heuristics)",
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    test_path = Path(args.test)
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    if not test_path.exists():
        print(f"Error: Test file not found: {test_path}")
        return 1
    
    # Load vocabulary and determine tagpos
    print(f"Loading vocabulary from {model_path}...")
    vocab = load_vocab(model_path)
    print(f"Loaded {len(vocab)} vocabulary entries")
    
    # Get tagpos from model
    tagpos = get_tagpos_from_model(model_path)
    print(f"Using tagpos: {tagpos}")
    
    # Load gold standard
    print(f"Loading gold standard from {test_path}...")
    gold_text = test_path.read_text(encoding="utf-8", errors="replace")
    gold_doc = conllu_to_document(gold_text, doc_id=test_path.stem)
    # Store sentence count immediately after loading (before any modifications)
    # Use direct len() call - gold_doc.sentences is always a list
    gold_sentence_count = len(gold_doc.sentences)
    
    # CRITICAL: Detag the input to avoid data leakage
    # The tagger preserves existing tags when no candidates are found,
    # which would make "unknown"/"fallback" sources appear 100% accurate
    print("Detagging input to prevent data leakage...")
    from flexipipe.check import _detag_document
    input_doc = _detag_document(gold_doc)
    
    # Debug: Check a sample token after detagging
    if args.debug or args.verbose:
        for sent in input_doc.sentences:
            if sent.tokens:
                sample = sent.tokens[0]
                print(f"Sample detagged token: form='{sample.form}', xpos='{sample.xpos}', "
                      f"upos='{sample.upos}', feats='{sample.feats}', lemma='{sample.lemma}'")
                # Also check if any tokens still have tags (shouldn't happen)
                tagged_count = 0
                for tok in sent.tokens[:10]:  # Check first 10 tokens
                    if (tok.xpos and tok.xpos != "_") or (tok.upos and tok.upos != "_"):
                        tagged_count += 1
                        if args.debug:
                            print(f"  WARNING: Token '{tok.form}' still has tags: xpos='{tok.xpos}', upos='{tok.upos}'")
                if tagged_count > 0:
                    print(f"  WARNING: {tagged_count} tokens in first sentence still have tags after detagging!")
                break
    
    # Tag with flexitag
    print("Tagging with flexitag...")
    # CRITICAL: Pass tagpos from model to tagger - this ensures we use the correct tag attribute
    # that the model was trained with (e.g., "upos" if model has tag_attribute="upos")
    flexitag_options = {"tagpos": tagpos, "overwrite": True}  # Force overwrite to prevent preserving any existing tags
    if args.debug:
        flexitag_options["debug"] = True
    if args.endlen is not None:
        flexitag_options["endlen"] = args.endlen
    fallback = FlexitagFallback(str(model_path), options=flexitag_options, debug=args.debug)
    
    # Measure total time (including Python overhead)
    import time
    total_start_time = time.time()
    pred_result = fallback.tag(input_doc)
    total_elapsed_time = time.time() - total_start_time
    pred_doc = pred_result.document
    
    # Debug: print timing immediately
    if args.verbose or args.debug:
        print(f"[debug] Total elapsed time: {total_elapsed_time:.3f} seconds")
        if hasattr(pred_result, 'stats') and pred_result.stats:
            print(f"[debug] C++ elapsed_seconds: {pred_result.stats.get('elapsed_seconds', 'N/A')}")
    
    # Debug: Check what we got (always print when verbose)
    if args.verbose or args.debug:
        print(f"[debug] pred_result.stats keys: {list(pred_result.stats.keys()) if hasattr(pred_result, 'stats') and pred_result.stats else 'N/A'}")
        print(f"[debug] pred_result.stats values: {pred_result.stats if hasattr(pred_result, 'stats') else 'N/A'}")
        print(f"[debug] pred_doc has {len(pred_doc.sentences) if pred_doc.sentences else 0} sentences")
        print(f"[debug] gold_doc has {len(gold_doc.sentences) if gold_doc.sentences else 0} sentences")
    
    # Debug: Check a sample token after tagging
    if args.debug or args.verbose:
        for sent in pred_doc.sentences:
            if sent.tokens:
                sample = sent.tokens[0]
                print(f"Sample tagged token: form='{sample.form}', xpos='{sample.xpos}', "
                      f"upos='{sample.upos}', feats='{sample.feats}', lemma='{sample.lemma}'")
                # Also check corresponding gold token
                for gold_sent in gold_doc.sentences:
                    if gold_sent.tokens and gold_sent.tokens[0].form == sample.form:
                        gold_sample = gold_sent.tokens[0]
                        print(f"Sample gold token: form='{gold_sample.form}', xpos='{gold_sample.xpos}', "
                              f"upos='{gold_sample.upos}', feats='{gold_sample.feats}', lemma='{gold_sample.lemma}'")
                        break
                break
    
    # Analyze errors
    print("Analyzing errors...")
    # Store token counts for reporting (orthographic tokens only)
    def count_ortho_tokens(doc: Document) -> int:
        count = 0
        for sent in doc.sentences:
            count += len(sent.tokens)  # Count orthographic tokens only
        return count
    gold_token_count = count_ortho_tokens(gold_doc)
    stats = analyze_errors(gold_doc, pred_doc, vocab, tagpos=tagpos, verbose=args.debug)
    stats["gold_token_count"] = gold_token_count  # Store for reporting
    # Ensure sentence count is set early (before stats might be modified)
    stats["sentence_count"] = gold_sentence_count
    # Store timing info immediately after analyze_errors (before any other modifications)
    stats["total_elapsed_seconds"] = float(total_elapsed_time)
    
    # Extract stats - pred_result.stats should be a dict from engine.py
    elapsed_seconds = 0.0
    word_count = gold_token_count
    
    # Extract stats from pred_result - it should always have stats
    if hasattr(pred_result, 'stats'):
        stats_dict = pred_result.stats
        if isinstance(stats_dict, dict) and stats_dict:
            # Get elapsed_seconds - it should always be present
            elapsed_val = stats_dict.get("elapsed_seconds")
            if elapsed_val is not None:
                elapsed_seconds = float(elapsed_val)
            word_count_val = stats_dict.get("word_count")
            if word_count_val is not None:
                word_count = int(word_count_val)
    
    # Use total_elapsed_time if C++ timing is not available or seems wrong
    # But prefer C++ timing if it's available and reasonable
    if elapsed_seconds == 0.0 or elapsed_seconds < 0.001:
        elapsed_seconds = total_elapsed_time
    
    # Store timing and sentence info in stats dict
    # Recalculate sentence count directly from gold_doc (it should still have sentences)
    # Always recalculate to ensure we have the correct value - gold_doc should still have sentences
    final_sentence_count = len(gold_doc.sentences) if hasattr(gold_doc, 'sentences') and gold_doc.sentences else gold_sentence_count
    
    # Force set the values - ensure they're actually stored (convert to proper types)
    # Store both C++ timing (from stats) and total timing (from Python)
    # Overwrite any existing values to ensure they're correct
    stats["elapsed_seconds"] = float(elapsed_seconds)  # C++ timing
    stats["total_elapsed_seconds"] = float(total_elapsed_time)  # Total Python timing (overwrite earlier value)
    stats["word_count"] = int(word_count) if word_count else int(gold_token_count)
    # Overwrite sentence_count to ensure it's correct
    stats["sentence_count"] = int(final_sentence_count)
    
    # Debug: verify values are stored (only when verbose)
    if args.verbose or args.debug:
        print(f"[debug] After storing: elapsed_seconds={stats.get('elapsed_seconds')}, total_elapsed_seconds={stats.get('total_elapsed_seconds')}, total_elapsed_time={total_elapsed_time}")
    
    # Calculate speeds using the actual elapsed time (prefer C++ timing if available)
    timing_for_speed = elapsed_seconds if elapsed_seconds > 0 else total_elapsed_time
    if timing_for_speed > 0:
        stats["speed"] = float(stats["word_count"]) / timing_for_speed
        stats["sent_speed"] = float(stats["sentence_count"]) / timing_for_speed
    else:
        stats["speed"] = 0.0
        stats["sent_speed"] = 0.0
    
    # Always print stats info when verbose
    if args.verbose or args.debug:
        print(f"[debug] pred_result.stats type: {type(pred_result.stats)}")
        print(f"[debug] pred_result.stats content: {pred_result.stats}")
        print(f"[debug] Extracted: elapsed={elapsed_seconds}, word_count={word_count}")
        print(f"[debug] gold_sentence_count={gold_sentence_count}, final_sentence_count={final_sentence_count}")
        print(f"[debug] stats dict keys: {list(stats.keys())[:10]}...")  # First 10 keys
    
    # Verify values are in stats before printing (only when verbose)
    if args.verbose or args.debug:
        print(f"[debug] Before print_report: sentence_count={stats.get('sentence_count')}, elapsed_seconds={stats.get('elapsed_seconds')}, total_elapsed_seconds={stats.get('total_elapsed_seconds')}, word_count={stats.get('word_count')}")
    
    # Print report
    print_report(stats, tagpos=tagpos, verbose=args.debug)
    
    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

