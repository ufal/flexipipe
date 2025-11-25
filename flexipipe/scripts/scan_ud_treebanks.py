#!/usr/bin/env python3
"""
Scan Universal Dependencies treebanks and populate the UD tags repository.

Usage:
    python -m flexipipe.scripts.scan_ud_treebanks --treebank-root /path/to/ud-treebanks
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set

from ..ud_tags_repository import (
    load_repository,
    save_repository,
    merge_treebank_data,
    STANDARD_UPOS,
    STANDARD_FEATS,
    STANDARD_MISC,
    DOCUMENT_FIELDS,
    SENTENCE_FIELDS,
)


MISC_FIELD_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _normalize_misc_field_name(raw_name: str) -> Optional[str]:
    """
    Normalize MISC field names to clean Field=Value keys.
    
    - Removes leading # characters: ###sent_id -> sent_id.
    - Removes trailing bracket/parenthesis annotations: Field[XYZ] -> Field, Field(88) -> Field, text[a9] -> text.
    - Removes trailing numbered suffixes: transl_ru-13 -> transl_ru (if pattern matches).
    - Replaces non-alphanumeric characters (except _) with underscores or removes them.
    - Accepts only ASCII alphanumeric names with underscores: [A-Za-z0-9_]+
    - Returns None for invalid or empty names.
    """
    if not raw_name:
        return None
    # Strip leading # characters (e.g., ###sent_id -> sent_id)
    cleaned = raw_name.lstrip('#').strip()
    if not cleaned:
        return None
    # Strip any bracketed or parenthesized suffix (e.g., SpaceAfter[BreakLevels], ptext(88), text[a9])
    # This removes everything from [ or ( to the end of the string
    cleaned = re.sub(r"[\[\(].*$", "", cleaned).strip()
    if not cleaned:
        return None
    # Strip trailing numbered suffixes like -13, _13, .13 (e.g., transl_ru-13 -> transl_ru)
    # But only if it's a clear numbered suffix pattern (ends with -digit, _digit, or .digit)
    cleaned = re.sub(r"[-_.]\d+$", "", cleaned).strip()
    if not cleaned:
        return None
    # Must be ASCII-only (reject non-ASCII like ṣa-ri-ri)
    try:
        cleaned.encode('ascii')
    except UnicodeEncodeError:
        return None
    # Replace any non-alphanumeric characters (except underscore) with underscores
    # This handles cases like "field-name" -> "field_name", "field.name" -> "field_name"
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", cleaned)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    if not cleaned:
        return None
    # Must match the strict pattern: only alphanumeric and underscores
    if not MISC_FIELD_PATTERN.match(cleaned):
        return None
    return cleaned


def scan_conllu_file(conllu_path: Path) -> tuple[
    Set[str],  # UPOS tags
    Dict[str, Dict[str, Set[str]]],  # FEATS: {feat_name: {upos: {values}}}
    Set[str],  # MISC fields
    Set[str],  # Document fields
    Set[str],  # Sentence fields
]:
    """Scan a single CoNLL-U file and extract all UD tags."""
    upos_tags: Set[str] = set()
    feats: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    misc_fields: Set[str] = set()
    doc_fields: Set[str] = set()
    sent_fields: Set[str] = set()
    
    current_doc_attrs: Set[str] = set()
    current_sent_attrs: Set[str] = set()
    
    try:
        with conllu_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    # End of sentence - reset sentence attrs
                    current_sent_attrs.clear()
                    continue
                
                if line.startswith("#"):
                    # Comment line
                    if line.startswith("# newdoc"):
                        # Document-level metadata
                        for part in line.split()[1:]:
                            if "=" in part:
                                key_raw = part.split("=", 1)[0]
                                # Normalize field name (strip brackets/parentheses)
                                key = _normalize_misc_field_name(key_raw) or key_raw
                                doc_fields.add(key)
                                current_doc_attrs.add(key)
                    elif line.startswith("# newpar"):
                        # Paragraph-level (tracked but not stored separately)
                        pass
                    elif "=" in line:
                        # Sentence-level or document-level metadata
                        # Format: # key = value or # key=value
                        match = re.match(r"#\s*([^=]+?)\s*=\s*(.+)", line)
                        if match:
                            key_raw = match.group(1).strip()
                            # Normalize field name (strip brackets/parentheses like text[a101] -> text)
                            key = _normalize_misc_field_name(key_raw) or key_raw
                            # Check if it's a known sentence field
                            if key in {"sent_id", "text", "lang", "date", "speaker", "participant", 
                                      "annotator", "translation", "align", "translation_lang", "corr"}:
                                sent_fields.add(key)
                                current_sent_attrs.add(key)
                            else:
                                # Could be document or sentence - track both
                                doc_fields.add(key)
                                sent_fields.add(key)
                    continue
                
                # Token line (10 columns)
                parts = line.split("\t")
                if len(parts) < 10:
                    continue
                
                # Column 4: UPOS
                upos = parts[3].strip()
                if upos and upos != "_":
                    upos_tags.add(upos)
                
                # Column 6: FEATS
                feats_str = parts[5].strip()
                if feats_str and feats_str != "_":
                    for feat_pair in feats_str.split("|"):
                        if "=" in feat_pair:
                            feat_name, feat_value = feat_pair.split("=", 1)
                            feat_name = feat_name.strip()
                            feat_value = feat_value.strip()
                            if feat_name and feat_value:
                                feats[feat_name][upos].add(feat_value)
                
                # Column 10: MISC
                misc_str = parts[9].strip()
                if misc_str and misc_str != "_":
                    for misc_part in misc_str.split("|"):
                        misc_part = misc_part.strip()
                        if not misc_part:
                            continue
                        # Only track clean Field=Value attributes
                        if "=" not in misc_part:
                            continue
                        misc_key_raw = misc_part.split("=", 1)[0].strip()
                        if not misc_key_raw:
                            continue
                        misc_key = _normalize_misc_field_name(misc_key_raw)
                        if misc_key:
                            misc_fields.add(misc_key)
    
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {conllu_path}: {e}", file=__import__("sys").stderr)
    
    return upos_tags, feats, misc_fields, doc_fields, sent_fields


def scan_treebank(treebank_path: Path) -> tuple[
    Set[str],
    Dict[str, Dict[str, Set[str]]],
    Set[str],
    Set[str],
    Set[str],
]:
    """Scan all CoNLL-U files in a treebank directory."""
    all_upos: Set[str] = set()
    all_feats: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    all_misc: Set[str] = set()
    all_doc_fields: Set[str] = set()
    all_sent_fields: Set[str] = set()
    
    # Look for .conllu files
    conllu_files = list(treebank_path.rglob("*.conllu"))
    if not conllu_files:
        # Also check for .conllu files in subdirectories
        conllu_files = list(treebank_path.glob("*.conllu"))
    
    for conllu_file in conllu_files:
        upos, feats, misc, doc_fields, sent_fields = scan_conllu_file(conllu_file)
        all_upos.update(upos)
        all_misc.update(misc)
        all_doc_fields.update(doc_fields)
        all_sent_fields.update(sent_fields)
        
        # Merge FEATS
        for feat_name, upos_values in feats.items():
            for upos, values in upos_values.items():
                all_feats[feat_name][upos].update(values)
    
    return all_upos, all_feats, all_misc, all_doc_fields, all_sent_fields


def _format_extra_column(items: list[str]) -> str:
    if not items:
        return ""
    preview = ", ".join(items[:4])
    if len(items) > 4:
        preview += f", +{len(items) - 4} more"
    return f"{len(items)}: {preview}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan UD treebanks and populate the UD tags repository"
    )
    parser.add_argument(
        "--treebank-root",
        type=Path,
        required=True,
        help="Root directory containing UD treebanks (e.g., /path/to/ud-treebanks-v2.15)",
    )
    parser.add_argument(
        "--treebank",
        help="Specific treebank to scan (e.g., UD_English-EWT). If not specified, scans all.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    
    args = parser.parse_args()
    
    treebank_root = args.treebank_root
    if not treebank_root.exists():
        print(f"Error: Treebank root directory does not exist: {treebank_root}")
        return 1
    
    # Load existing repository
    repo = load_repository()
    
    # Find treebanks to scan
    if args.treebank:
        treebank_dirs = [treebank_root / args.treebank]
        treebank_dirs = [d for d in treebank_dirs if d.exists()]
        if not treebank_dirs:
            print(f"Error: Treebank not found: {args.treebank}")
            return 1
    else:
        # Scan all UD_* directories
        treebank_dirs = sorted([d for d in treebank_root.iterdir() if d.is_dir() and d.name.startswith("UD_")])
        if not treebank_dirs:
            print(f"Warning: No UD_* directories found in {treebank_root}")
            return 1
    
    print(f"Scanning {len(treebank_dirs)} treebank(s)...")
    
    extras_report: list[Dict[str, list[str]]] = []

    for treebank_dir in treebank_dirs:
        treebank_name = treebank_dir.name
        if args.verbose:
            print(f"Scanning {treebank_name}...")
        
        try:
            upos, feats, misc, doc_fields, sent_fields = scan_treebank(treebank_dir)
            
            if args.verbose:
                print(f"  Found {len(upos)} UPOS tags, {len(feats)} FEATS, {len(misc)} MISC fields")
                
                # Show non-standard fields
                non_std_misc = sorted(m for m in misc if m not in STANDARD_MISC)
                if non_std_misc:
                    print(f"  MISC fields: {', '.join(non_std_misc)}")
                
                # Exclude standard sentence fields (id, text, sent_id)
                standard_sent_excluded = {"id", "text", "sent_id"}
                non_std_sent = sorted(f for f in sent_fields if f not in SENTENCE_FIELDS and f not in standard_sent_excluded)
                if non_std_sent:
                    print(f"  SENT features: {', '.join(non_std_sent)}")
                
                # Exclude standard document fields (id, newdoc_id)
                standard_doc_excluded = {"id", "newdoc_id"}
                non_std_doc = sorted(f for f in doc_fields if f not in DOCUMENT_FIELDS and f not in standard_doc_excluded)
                if non_std_doc:
                    print(f"  P/TEXT features: {', '.join(non_std_doc)}")
            
            merge_treebank_data(
                repo,
                treebank_name,
                upos,
                feats,
                misc,
                doc_fields,
                sent_fields,
            )

            extras_report.append(
                {
                    "treebank": treebank_name,
                    "upos": sorted(u for u in upos if u not in STANDARD_UPOS),
                    "feats": sorted(f for f in feats.keys() if f not in STANDARD_FEATS),
                    "misc": sorted(m for m in misc if m not in STANDARD_MISC),
                    "doc_fields": sorted(field for field in doc_fields if field not in DOCUMENT_FIELDS),
                    "sent_fields": sorted(field for field in sent_fields if field not in SENTENCE_FIELDS),
                }
            )
        except Exception as e:
            print(f"Error scanning {treebank_name}: {e}", file=__import__("sys").stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Save updated repository
    from datetime import datetime
    repo["last_updated"] = datetime.now().isoformat()
    save_repository(repo)
    
    print(f"\nRepository updated:")
    print(f"  UPOS tags: {len(repo['upos']['standard'])} standard, {len(repo['upos']['extended'])} extended")
    print(f"  FEATS: {len(repo['feats']['standard'])} standard, {len(repo['feats']['extended'])} extended")
    print(f"  MISC fields: {len(repo['misc']['standard'])} standard, {len(repo['misc']['extended'])} extended")
    print(f"  Treebanks scanned: {len(repo['treebanks_scanned'])}")

    # Print extras table
    extras_with_data = [
        entry for entry in extras_report
        if entry["upos"] or entry["feats"] or entry["misc"] or entry["doc_fields"] or entry["sent_fields"]
    ]
    if extras_with_data:
        print("\nTreebank extras summary:")
        print(f"{'Treebank':<30} {'UPOS':<20} {'FEATS':<20} {'MISC':<25} {'DocFields':<15} {'SentFields':<15}")
        print("-" * 130)
        for entry in extras_with_data:
            print(
                f"{entry['treebank']:<30} "
                f"{_format_extra_column(entry['upos']):<20} "
                f"{_format_extra_column(entry['feats']):<20} "
                f"{_format_extra_column(entry['misc']):<25} "
                f"{_format_extra_column(entry['doc_fields']):<15} "
                f"{_format_extra_column(entry['sent_fields']):<15}"
            )
    else:
        print("\nNo treebanks introduced non-standard tags or fields.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

