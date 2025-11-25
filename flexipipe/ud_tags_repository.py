"""
Universal Dependencies tags repository.

Maintains a comprehensive catalog of all UD tags at all levels:
- UPOS tags (standard and extended)
- FEATS (with UPOS associations)
- MISC fields (from treebanks and defined)
- Sentence/Document-level fields

For each field, tracks:
- Which Doc element it's stored in
- How it's printed/serialized from Doc (can differ, e.g., Norm vs Normalization)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Standard UD UPOS tags
STANDARD_UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}

# Extended UPOS tags (seen in treebanks but not in standard)
EXTENDED_UPOS: Set[str] = set()

# Standard UD FEATS (from UD guidelines)
STANDARD_FEATS: Dict[str, Dict[str, Any]] = {
    "Abbr": {"values": ["Yes"], "upos": ["ADJ", "ADP", "ADV", "NOUN", "PROPN", "VERB"]},
    "Animacy": {"values": ["Anim", "Inan", "Hum", "Nhum"], "upos": ["NOUN", "PRON", "PROPN"]},
    "Aspect": {"values": ["Imp", "Perf", "Prog", "Prosp"], "upos": ["VERB", "AUX"]},
    "Case": {"values": ["Abs", "Acc", "Erg", "Nom", "Dat", "Gen", "Loc", "Ins", "Voc", "Abe", "Ben", "Cau", "Cmp", "Cns", "Com", "Dis", "Equ", "Ess", "Par", "Tem", "Tra"], "upos": ["ADJ", "DET", "NOUN", "NUM", "PRON", "PROPN"]},
    "Definite": {"values": ["Def", "Ind", "Spec", "Cons", "2"], "upos": ["DET", "PRON"]},
    "Degree": {"values": ["Pos", "Cmp", "Sup", "Abs"], "upos": ["ADJ", "ADV"]},
    "Evident": {"values": ["Fh", "Nfh"], "upos": ["VERB", "AUX"]},
    "Foreign": {"values": ["Yes"], "upos": ["X"]},
    "Gender": {"values": ["Masc", "Fem", "Neut", "Com", "Com,Neut"], "upos": ["ADJ", "DET", "NOUN", "PRON", "PROPN", "VERB"]},
    "Mood": {"values": ["Ind", "Imp", "Cnd", "Pot", "Sub", "Jus", "Nec", "Opt", "Qot", "Des"], "upos": ["VERB", "AUX"]},
    "NounClass": {"values": ["Bantu1", "Bantu2", "Bantu3", "Bantu4", "Bantu5", "Bantu6", "Bantu7", "Bantu8", "Bantu9", "Bantu10", "Bantu11", "Bantu12", "Bantu13", "Bantu14", "Bantu15", "Bantu16", "Bantu17", "Bantu18", "Bantu19", "Bantu20", "Bantu21", "Bantu22", "Bantu23"], "upos": ["NOUN", "PRON", "PROPN"]},
    "Number": {"values": ["Sing", "Plur", "Dual", "Tri", "Pauc", "Grpa", "Grpl", "Count"], "upos": ["ADJ", "DET", "NOUN", "PRON", "PROPN", "VERB"]},
    "NumType": {"values": ["Card", "Ord", "Mult", "Frac", "Sets", "Dist", "Range"], "upos": ["ADJ", "DET", "NUM", "PRON"]},
    "Person": {"values": ["1", "2", "3", "4"], "upos": ["PRON", "VERB", "AUX", "DET"]},
    "Polarity": {"values": ["Pos", "Neg"], "upos": ["ADJ", "ADV", "VERB", "AUX", "DET", "INTJ", "NOUN", "PART", "PRON"]},
    "Polite": {"values": ["Elev", "Form", "Humb", "Infm"], "upos": ["PRON", "VERB", "AUX"]},
    "Poss": {"values": ["Yes"], "upos": ["ADJ", "DET", "PRON"]},
    "PronType": {"values": ["Art", "Dem", "Emp", "Exc", "Ind", "Int", "Neg", "Prs", "Rcp", "Rel", "Tot"], "upos": ["ADJ", "DET", "PRON"]},
    "Reflex": {"values": ["Yes"], "upos": ["PRON", "DET"]},
    "Tense": {"values": ["Past", "Pres", "Fut", "Imp", "Pqp"], "upos": ["VERB", "AUX"]},
    "Typo": {"values": ["Yes"], "upos": ["X"]},
    "VerbForm": {"values": ["Fin", "Inf", "Sup", "Part", "Trans", "Gdv", "Conv", "Vnoun"], "upos": ["VERB", "AUX", "ADJ"]},
    "Voice": {"values": ["Act", "Pass", "Mid", "Rcp", "Antip", "Cau", "Dir", "Inv", "Rfl"], "upos": ["VERB", "AUX"]},
}

# Extended FEATS (seen in treebanks but not in standard)
EXTENDED_FEATS: Dict[str, Dict[str, Any]] = {}

# Standard MISC fields (from UD guidelines and common usage)
STANDARD_MISC: Dict[str, Dict[str, Any]] = {
    "SpaceAfter": {"description": "Whether there is a space after this token", "doc_field": "space_after", "doc_type": "Token|SubToken", "print_as": "SpaceAfter"},
    "Normalization": {"description": "Normalized form of the token", "doc_field": "reg", "doc_type": "Token|SubToken", "print_as": "Normalization"},
    "ModernForm": {"description": "Modern form of the token", "doc_field": "mod", "doc_type": "Token|SubToken", "print_as": "ModernForm"},
    "Expansion": {"description": "Expanded form of the token", "doc_field": "expan", "doc_type": "Token|SubToken", "print_as": "Expansion"},
    "Translit": {"description": "Transliteration of the token", "doc_field": "trslit", "doc_type": "Token|SubToken", "print_as": "Translit"},
    "LTransLit": {"description": "Lemma transliteration", "doc_field": "ltrslit", "doc_type": "Token|SubToken", "print_as": "LTransLit"},
    "TokId": {"description": "Token identifier", "doc_field": "tokid", "doc_type": "Token|SubToken", "print_as": "TokId"},
    "Entity": {"description": "Named entity label (IOB format)", "doc_field": "sentence.entities", "doc_type": "Sentence", "print_as": "Entity"},
    "NE": {"description": "Named entity label (NE format)", "doc_field": "sentence.entities", "doc_type": "Sentence", "print_as": "NE"},
    "Corr": {"description": "Corrected form", "doc_field": "corr", "doc_type": "Token|SubToken", "print_as": "Corr"},
    "Lex": {"description": "Lexical form", "doc_field": "lex", "doc_type": "Token|SubToken", "print_as": "Lex"},
}

# Extended MISC fields (from treebanks)
EXTENDED_MISC: Dict[str, Dict[str, Any]] = {}

# Document-level fields (from UD guidelines)
DOCUMENT_FIELDS: Dict[str, Dict[str, Any]] = {
    "newdoc_id": {"description": "Unique document identifier", "doc_field": "id", "doc_type": "Document", "print_as": "newdoc_id"},
    "title": {"description": "Document title", "doc_field": "attrs.title", "doc_type": "Document", "print_as": "title"},
    "author": {"description": "Document author", "doc_field": "attrs.author", "doc_type": "Document", "print_as": "author"},
    "date": {"description": "Publication/creation date", "doc_field": "attrs.date", "doc_type": "Document", "print_as": "date"},
    "genre": {"description": "Text genre", "doc_field": "attrs.genre", "doc_type": "Document", "print_as": "genre"},
    "publisher": {"description": "Publisher information", "doc_field": "attrs.publisher", "doc_type": "Document", "print_as": "publisher"},
    "url": {"description": "Source URL", "doc_field": "attrs.url", "doc_type": "Document", "print_as": "url"},
    "license": {"description": "Licensing information", "doc_field": "attrs.license", "doc_type": "Document", "print_as": "license"},
    "source": {"description": "Original source description", "doc_field": "attrs.source", "doc_type": "Document", "print_as": "source"},
}

# Sentence-level fields (from UD guidelines)
SENTENCE_FIELDS: Dict[str, Dict[str, Any]] = {
    "sent_id": {"description": "Unique sentence identifier", "doc_field": "sent_id", "doc_type": "Sentence", "print_as": "sent_id"},
    "text": {"description": "Original sentence text", "doc_field": "text", "doc_type": "Sentence", "print_as": "text"},
    "lang": {"description": "Language code (ISO 639-1)", "doc_field": "attrs.lang", "doc_type": "Sentence", "print_as": "lang"},
    "date": {"description": "Sentence-specific timestamp", "doc_field": "attrs.date", "doc_type": "Sentence", "print_as": "date"},
    "speaker": {"description": "For spoken texts", "doc_field": "attrs.speaker", "doc_type": "Sentence", "print_as": "speaker"},
    "participant": {"description": "Conversation participant", "doc_field": "attrs.participant", "doc_type": "Sentence", "print_as": "participant"},
    "annotator": {"description": "Annotation information", "doc_field": "attrs.annotator", "doc_type": "Sentence", "print_as": "annotator"},
    "translation": {"description": "Translation", "doc_field": "attrs.translation", "doc_type": "Sentence", "print_as": "translation"},
    "align": {"description": "Alignment", "doc_field": "attrs.align", "doc_type": "Sentence", "print_as": "align"},
    "translation_lang": {"description": "Translation language", "doc_field": "attrs.translation_lang", "doc_type": "Sentence", "print_as": "translation_lang"},
    "corr": {"description": "Corrected sentence", "doc_field": "corr", "doc_type": "Sentence", "print_as": "corr"},
}


def get_repository_path() -> Path:
    """Get the path to the UD tags repository JSON file."""
    from .model_storage import get_flexipipe_models_dir
    models_dir = get_flexipipe_models_dir(create=True)
    return models_dir / "ud_tags_repository.json"


def load_repository() -> Dict[str, Any]:
    """Load the UD tags repository from disk."""
    repo_path = get_repository_path()
    if not repo_path.exists():
        return _create_empty_repository()
    
    try:
        with repo_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return _create_empty_repository()


def save_repository(repo: Dict[str, Any]) -> None:
    """Save the UD tags repository to disk."""
    repo_path = get_repository_path()
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    with repo_path.open("w", encoding="utf-8") as f:
        json.dump(repo, f, indent=2, ensure_ascii=False, sort_keys=True)


def _create_empty_repository() -> Dict[str, Any]:
    """Create an empty repository structure."""
    return {
        "version": "1.0",
        "upos": {
            "standard": sorted(list(STANDARD_UPOS)),
            "extended": sorted(list(EXTENDED_UPOS)),
            "usage": {},  # {upos: {treebank: count}}
        },
        "feats": {
            "standard": STANDARD_FEATS,
            "extended": EXTENDED_FEATS,
            "usage": {},  # {feat: {upos: {treebank: count}}}
        },
        "misc": {
            "standard": STANDARD_MISC,
            "extended": EXTENDED_MISC,
            "usage": {},  # {misc_field: {treebank: count}}
        },
        "document_fields": DOCUMENT_FIELDS,
        "sentence_fields": SENTENCE_FIELDS,
        "treebanks_scanned": [],
        "last_updated": None,
    }


def merge_treebank_data(
    repo: Dict[str, Any],
    treebank_name: str,
    upos_tags: Set[str],
    feats: Dict[str, Dict[str, Set[str]]],  # {feat_name: {upos: {values}}}
    misc_fields: Set[str],
    doc_fields: Set[str],
    sent_fields: Set[str],
) -> None:
    """Merge data from a scanned treebank into the repository."""
    # Update UPOS
    for upos in upos_tags:
        if upos not in STANDARD_UPOS:
            if upos not in repo["upos"]["extended"]:
                repo["upos"]["extended"].append(upos)
            repo["upos"]["extended"].sort()
        
        if upos not in repo["upos"]["usage"]:
            repo["upos"]["usage"][upos] = {}
        repo["upos"]["usage"][upos][treebank_name] = repo["upos"]["usage"][upos].get(treebank_name, 0) + 1
    
    # Update FEATS
    for feat_name, upos_values in feats.items():
        if feat_name not in repo["feats"]["standard"]:
            if feat_name not in repo["feats"]["extended"]:
                repo["feats"]["extended"][feat_name] = {
                    "values": [],
                    "upos": [],
                    "description": f"Extended feature from treebanks",
                }
            # Update values and UPOS associations
            for upos, values in upos_values.items():
                if upos not in repo["feats"]["extended"][feat_name]["upos"]:
                    repo["feats"]["extended"][feat_name]["upos"].append(upos)
                for value in values:
                    if value not in repo["feats"]["extended"][feat_name]["values"]:
                        repo["feats"]["extended"][feat_name]["values"].append(value)
        
        # Track usage
        if feat_name not in repo["feats"]["usage"]:
            repo["feats"]["usage"][feat_name] = {}
        for upos in upos_values.keys():
            if upos not in repo["feats"]["usage"][feat_name]:
                repo["feats"]["usage"][feat_name][upos] = {}
            repo["feats"]["usage"][feat_name][upos][treebank_name] = (
                repo["feats"]["usage"][feat_name][upos].get(treebank_name, 0) + 1
            )
    
    # Update MISC fields
    for misc_field in misc_fields:
        if misc_field not in repo["misc"]["standard"]:
            if misc_field not in repo["misc"]["extended"]:
                repo["misc"]["extended"][misc_field] = {
                    "description": f"Extended MISC field from treebanks",
                    "doc_field": None,  # Unknown mapping
                    "doc_type": "Token|SubToken",
                    "print_as": misc_field,
                }
        
        if misc_field not in repo["misc"]["usage"]:
            repo["misc"]["usage"][misc_field] = {}
        repo["misc"]["usage"][misc_field][treebank_name] = (
            repo["misc"]["usage"][misc_field].get(treebank_name, 0) + 1
        )
    
    # Track document and sentence fields (for completeness)
    # These are usually standard, but we track usage
    for field in doc_fields:
        if field not in repo["document_fields"]:
            repo["document_fields"][field] = {
                "description": f"Extended document field from treebanks",
                "doc_field": f"attrs.{field}",
                "doc_type": "Document",
                "print_as": field,
            }
    
    for field in sent_fields:
        if field not in repo["sentence_fields"]:
            repo["sentence_fields"][field] = {
                "description": f"Extended sentence field from treebanks",
                "doc_field": f"attrs.{field}",
                "doc_type": "Sentence",
                "print_as": field,
            }
    
    # Track scanned treebank
    if treebank_name not in repo["treebanks_scanned"]:
        repo["treebanks_scanned"].append(treebank_name)
        repo["treebanks_scanned"].sort()

