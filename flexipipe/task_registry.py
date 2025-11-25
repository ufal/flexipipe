"""Central definitions for flexipipe NLP tasks."""

from __future__ import annotations

from typing import Dict, Set

# Canonical task descriptions
TASK_DESCRIPTIONS: Dict[str, str] = {
    "segment": "Sentence segmentation.",
    "tokenize": "Tokenization / word segmentation.",
    "lemmatize": "Lemma generation.",
    "tag": "Universal POS tagging.",
    "xpos": "Language-specific POS tagging.",
    "parse": "Dependency parsing.",
    "normalize": "Text normalization / reg attribute generation.",
    "ner": "Named entity recognition.",
    "wsd": "Word-sense or frame disambiguation.",
    "gec": "Grammatical error detection/correction.",
    "topic": "Topic or document classification.",
    "langdetect": "Language identification.",
}

# Tasks enabled by default when --tasks is omitted
TASK_DEFAULTS = ["segment", "tokenize", "lemmatize", "tag", "parse", "normalize", "ner"]

# Tasks that must always run (pipeline requires them)
TASK_MANDATORY = {"segment", "tokenize"}

# Canonical task -> alias strings
_TASK_ALIAS_DEFINITIONS: Dict[str, Set[str]] = {
    "segment": {"segment", "segmentation", "sentence", "sent"},
    "tokenize": {"tokenize", "tokenization", "token"},
    "lemmatize": {"lemmatize", "lemmatization", "lemma"},
    "tag": {"tag", "upos", "pos"},
    "xpos": {"xpos"},
    "parse": {"parse", "parser", "depparse", "dependency"},
    "normalize": {"normalize", "normalization", "norm", "reg"},
    "ner": {"ner", "entity", "entities", "name", "named"},
    "wsd": {"wsd", "wsd-frame", "frame", "frames"},
    "gec": {"gec", "grammar", "grammatical", "error", "correction"},
    "topic": {"topic", "topics", "topic-model", "classification"},
    "langdetect": {"langdetect", "langid", "language-detection", "languageid"},
}

# Normalized alias lookup
TASK_ALIASES: Dict[str, Set[str]] = {}
TASK_LOOKUP: Dict[str, str] = {}

for canonical, aliases in _TASK_ALIAS_DEFINITIONS.items():
    normalized_aliases = {canonical.lower()}
    normalized_aliases.update(alias.lower() for alias in aliases)
    TASK_ALIASES[canonical] = normalized_aliases
    for alias in normalized_aliases:
        TASK_LOOKUP[alias] = canonical

