"""
Lexicon conversion utilities for FlexiPipe.

Converts external lexicons (UniMorph, etc.) to FlexiPipe vocabulary format.
Supports tagset.xml for mapping XPOS to UPOS+FEATS.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .tagset import parse_teitok_tagset, xpos_to_upos_feats


def load_unimorph_lexicon(
    file_path: Path,
    tagset_file: Optional[Path] = None,
    corpus_xpos_tags: Optional[Dict[str, int]] = None,
    default_count: int = 1,
) -> Dict:
    """
    Load a UniMorph lexicon file and convert it to FlexiPipe vocabulary format.
    
    UniMorph format is tab-separated: lemma\tform\tfeatures
    Where features are in UniMorph format (e.g., "V;IND;SG;3;PRS" for verb, indicative, singular, 3rd person, present).
    Note: The lemma comes FIRST in UniMorph format, not the form.
    
    If a tagset.xml file is provided and corpus_xpos_tags is provided, the system will:
    1. Map UniMorph features to UPOS+FEATS
    2. Find the best matching XPOS tag from the corpus for each entry
    3. Use the tagset to map XPOS -> UPOS+FEATS and verify/complete the mapping
    
    Args:
        file_path: Path to UniMorph lexicon file (tab-separated: lemma\tform\tfeatures)
        tagset_file: Optional path to tagset.xml file for XPOS mapping
        corpus_xpos_tags: Optional dict of {xpos: count} from corpus to find best matches
        default_count: Default count to use for entries (default: 1, since UniMorph doesn't have counts)
    
    Returns:
        Dictionary in FlexiPipe vocabulary format: {form: {upos, feats, lemma, count, xpos?} or [{...}, ...]}
    """
    vocab = defaultdict(list)
    tagset_def = None
    
    # Load tagset if provided
    if tagset_file and tagset_file.exists():
        tagset_def = parse_teitok_tagset(tagset_file)
    
    # Build reverse mapping: (upos, feats) -> [(xpos, count), ...] sorted by frequency
    xpos_candidates: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
    if corpus_xpos_tags and tagset_def:
        for xpos, count in corpus_xpos_tags.items():
            upos, feats = xpos_to_upos_feats(xpos, tagset_def)
            if upos != '_' and feats != '_':
                xpos_candidates[(upos, feats)].append((xpos, count))
        # Sort by frequency (descending)
        for key in xpos_candidates:
            xpos_candidates[key].sort(key=lambda x: x[1], reverse=True)
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            # UniMorph format is: lemma\tform\tfeatures
            lemma = parts[0].strip()
            form = parts[1].strip()
            
            if not form or not lemma:
                continue
            
            # Parse UniMorph features (if present)
            features = parts[2].strip() if len(parts) > 2 else ''
            
            # Convert UniMorph features to UD FEATS format
            upos, feats = _unimorph_to_ud(features)
            
            # If we have tagset and corpus tags, try to find best matching XPOS
            xpos = None
            if tagset_def and corpus_xpos_tags and upos != '_' and feats != '_':
                # Find best matching XPOS from corpus
                candidates = xpos_candidates.get((upos, feats), [])
                if candidates:
                    # Use the most frequent XPOS that matches
                    xpos = candidates[0][0]
                else:
                    # Try partial matching: find XPOS where all features in feats are present
                    # (allowing for additional features in XPOS)
                    best_match = _find_partial_feat_match(upos, feats, corpus_xpos_tags, tagset_def)
                    if best_match:
                        xpos = best_match
            
            # Create vocabulary entry
            entry: Dict[str, any] = {
                'lemma': lemma.lower(),
                'count': default_count
            }
            
            if upos != '_':
                entry['upos'] = upos
            if feats != '_':
                entry['feats'] = feats
            if xpos:
                entry['xpos'] = xpos
            
            # Add to vocabulary (handle multiple analyses for same form)
            vocab[form].append(entry)
    
    # Convert lists to single dict if only one analysis, otherwise keep as list
    result = {}
    for form, entries in vocab.items():
        if len(entries) == 1:
            result[form] = entries[0]
        else:
            result[form] = entries
    
    return result


def _unimorph_to_ud(features: str) -> Tuple[str, str]:
    """
    Convert UniMorph features to UD UPOS and FEATS.
    
    Args:
        features: UniMorph feature string (e.g., "V;IND;SG;3;PRS")
        
    Returns:
        Tuple of (upos, feats) in UD format
    """
    if not features:
        return '_', '_'
    
    # Basic mapping from UniMorph feature abbreviations to UD
    unimorph_parts = [f.strip() for f in features.split(';') if f.strip()]
    ud_feats = []
    upos = '_'
    
    # Map common UniMorph POS tags to UPOS
    pos_mapping = {
        'N': 'NOUN', 'V': 'VERB', 'ADJ': 'ADJ', 'ADV': 'ADV',
        'PRON': 'PRON', 'DET': 'DET', 'PREP': 'ADP', 'ADP': 'ADP',
        'CONJ': 'CCONJ', 'SCONJ': 'SCONJ', 'NUM': 'NUM', 'PUNCT': 'PUNCT'
    }
    
    # Map common UniMorph features to UD FEATS
    feature_mapping = {
        # Tense
        'PST': 'Tense=Past', 'PRS': 'Tense=Pres', 'FUT': 'Tense=Fut',
        # Person
        '1': 'Person=1', '2': 'Person=2', '3': 'Person=3',
        # Number
        'SG': 'Number=Sing', 'PL': 'Number=Plur',
        # Gender
        'MASC': 'Gender=Masc', 'FEM': 'Gender=Fem', 'NEUT': 'Gender=Neut',
        # Case
        'NOM': 'Case=Nom', 'ACC': 'Case=Acc', 'GEN': 'Case=Gen', 'DAT': 'Case=Dat',
        # Mood
        'IND': 'Mood=Ind', 'SUB': 'Mood=Sub', 'IMP': 'Mood=Imp',
        # Aspect
        'IPFV': 'Aspect=Imp', 'PFV': 'Aspect=Perf',
        # Voice
        'ACT': 'Voice=Act', 'PASS': 'Voice=Pass',
    }
    
    for part in unimorph_parts:
        # Check if it's a POS tag
        if part in pos_mapping:
            upos = pos_mapping[part]
        # Check if it's a feature
        elif part in feature_mapping:
            ud_feats.append(feature_mapping[part])
    
    if ud_feats:
        feats = '|'.join(sorted(ud_feats))
    else:
        feats = '_'
    
    return upos, feats


def _find_partial_feat_match(
    upos: str,
    feats: str,
    corpus_xpos_tags: Dict[str, int],
    tagset_def: Dict,
) -> Optional[str]:
    """
    Find the best matching XPOS tag from corpus that has all features in feats (partial matching).
    
    This allows for XPOS tags that have additional features beyond what's in the UniMorph entry.
    For example, if UniMorph has "Gender=Masc|Number=Sing", we can match an XPOS that has
    "Gender=Masc|Number=Sing|Case=Nom" (allowing the extra Case feature).
    
    Args:
        upos: UPOS tag
        feats: FEATS string in UD format (e.g., "Gender=Masc|Number=Sing")
        corpus_xpos_tags: Dict of {xpos: count} from corpus
        tagset_def: Tagset definition for XPOS -> UPOS+FEATS mapping
        
    Returns:
        Best matching XPOS tag, or None if no match found
    """
    if not feats or feats == '_':
        return None
    
    # Parse target features
    target_feats = {}
    for feat_pair in feats.split('|'):
        if '=' in feat_pair:
            key, value = feat_pair.split('=', 1)
            target_feats[key] = value
    
    if not target_feats:
        return None
    
    # Find all XPOS tags that match UPOS and have all target features
    candidates: List[Tuple[str, int]] = []
    
    for xpos, count in corpus_xpos_tags.items():
        xpos_upos, xpos_feats = xpos_to_upos_feats(xpos, tagset_def)
        
        # Must match UPOS
        if xpos_upos != upos:
            continue
        
        # Parse XPOS features
        xpos_feats_dict = {}
        if xpos_feats and xpos_feats != '_':
            for feat_pair in xpos_feats.split('|'):
                if '=' in feat_pair:
                    key, value = feat_pair.split('=', 1)
                    xpos_feats_dict[key] = value
        
        # Check if all target features are present in XPOS (partial match)
        if all(xpos_feats_dict.get(k) == v for k, v in target_feats.items()):
            candidates.append((xpos, count))
    
    if not candidates:
        return None
    
    # Return most frequent match
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def convert_lexicon_to_vocab(
    lexicon_file: Path,
    output_file: Path,
    *,
    tagset_file: Optional[Path] = None,
    corpus_file: Optional[Path] = None,
    default_count: int = 1,
) -> None:
    """
    Convert an external lexicon (UniMorph, etc.) to FlexiPipe vocabulary JSON format.
    
    Args:
        lexicon_file: Path to lexicon file (UniMorph format: lemma\tform\tfeatures)
        output_file: Path to output vocabulary JSON file
        tagset_file: Optional path to tagset.xml for XPOS mapping
        corpus_file: Optional path to corpus file (CoNLL-U or TEITOK XML) to extract XPOS tags
        default_count: Default count for entries
    """
    # Extract XPOS tags from corpus if provided
    corpus_xpos_tags: Optional[Dict[str, int]] = None
    if corpus_file:
        corpus_xpos_tags = _extract_xpos_tags(corpus_file)
    
    # Load and convert lexicon
    vocab = load_unimorph_lexicon(
        lexicon_file,
        tagset_file=tagset_file,
        corpus_xpos_tags=corpus_xpos_tags,
        default_count=default_count,
    )
    
    # Write vocabulary JSON
    output_data = {
        'vocab': vocab,
        'metadata': {
            'source': 'lexicon_conversion',
            'lexicon_file': str(lexicon_file),
            'tagset_file': str(tagset_file) if tagset_file else None,
            'corpus_file': str(corpus_file) if corpus_file else None,
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def _extract_xpos_tags(corpus_file: Path) -> Dict[str, int]:
    """
    Extract XPOS tags and their frequencies from a corpus file.
    
    Supports CoNLL-U and TEITOK XML formats.
    
    Args:
        corpus_file: Path to corpus file
        
    Returns:
        Dict of {xpos: count}
    """
    xpos_counts: Dict[str, int] = defaultdict(int)
    
    if corpus_file.suffix.lower() == '.conllu':
        # Parse CoNLL-U
        with open(corpus_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    xpos = parts[3]
                    if xpos and xpos != '_':
                        xpos_counts[xpos] += 1
    else:
        # Assume TEITOK XML
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(corpus_file)
            root = tree.getroot()
            
            # Find all tok elements with xpos attribute
            for tok in root.iter():
                if tok.tag.endswith('tok') or 'tok' in tok.tag.lower():
                    xpos = tok.get('xpos') or tok.get('msd') or tok.get('pos')
                    if xpos and xpos != '_':
                        xpos_counts[xpos] += 1
        except Exception:
            pass
    
    return dict(xpos_counts)

