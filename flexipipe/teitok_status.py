"""What a TEITOK document contains, and what a flexipipe run on it would do (`flexipipe info
document --input FILE`): for the TEITOK module's status page and its check before a run.

- contents: tokens, sentences, which annotation attributes are filled (in the corpus's attribute
  names: settings.xml mappings) and on how many tokens, named entities, the <change> history;
- the run: language (teiHeader, else settings.xml, else detected at run time), backend/model (the
  corpus's setting for that language, else flexipipe's choice), tasks (given, else the backend's
  defaults, + the mandatory ones), and what it would add or overwrite — existing values of the
  attributes those tasks write (unless kept: --keep), <s> elements added to a file that has none;
- whether the file can be written (the user running this: for TEITOK the web server).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from lxml import etree

# which annotation each task writes (internal names; mapped to the corpus's attribute names)
TASK_ATTRS: Dict[str, List[str]] = {
    "lemmatize": ["lemma"],
    "tag": ["upos", "xpos", "feats"],
    "xpos": ["xpos"],
    "parse": ["head", "deprel"],
    "normalize": ["reg"],
    "pron": ["pron"],
}
ANNOTATION_ATTRS = ["lemma", "upos", "xpos", "feats", "head", "deprel", "reg", "expan", "pron"]


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _document_language(root) -> Optional[str]:
    """teiHeader/profileDesc/langUsage/language/@ident (or its text), as `process` reads it."""
    for el in root.iter():
        if _local(el.tag) == "language" and _local(el.getparent().tag if el.getparent() is not None else "") == "langUsage":
            value = (el.get("ident") or (el.text or "")).strip()
            if value:
                return value
    return None


def has_tokens_without_sentences(path) -> bool:
    """The file has <tok> but no <s> (a TEITOK file tokenized but not split into sentences)."""
    has_tok = has_s = False
    try:
        for _, el in etree.iterparse(str(path), events=("start",), huge_tree=True, resolve_entities=False, no_network=True):
            name = _local(el.tag)
            if name == "s":
                has_s = True
                break
            if name == "tok":
                has_tok = True
    except (OSError, etree.XMLSyntaxError):
        return False
    return has_tok and not has_s


def document_language(path) -> Optional[str]:
    """A TEITOK file's language as `process` uses it: the teiHeader's, else settings.xml's default."""
    from .teitok_settings import TeitokSettings, find_settings_xml

    try:
        root = etree.parse(str(path), etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)).getroot()
        lang = _document_language(root)
    except (OSError, etree.XMLSyntaxError):
        lang = None
    if lang:
        return lang
    settings_path = find_settings_xml(Path(path))
    if settings_path and Path(settings_path).exists():
        return TeitokSettings.load(Path(settings_path)).default_language
    return None


def _xml_name(settings, internal: str) -> str:
    if settings is None:
        return internal
    name = settings.resolve_xml_attribute(internal, default=internal)
    return name or internal


def document_status(
    path: str,
    tasks: Optional[str] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    keep: Optional[str] = None,
    language: Optional[str] = None,
    sentence_layout: Optional[str] = None,
    task_backend: Optional[list] = None,
) -> dict:
    from .task_registry import TASK_MANDATORY
    from .teitok_settings import TeitokSettings, find_settings_xml

    file = Path(path)
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)
    root = etree.parse(str(file), parser).getroot()

    settings = None
    settings_path = find_settings_xml(file)
    if settings_path and Path(settings_path).exists():
        settings = TeitokSettings.load(Path(settings_path))

    if settings is not None:  # the names the corpus uses, as a run writes them (teitok_names.py)
        from .corpus_models import corpus_root as _corpus_root
        from .teitok_names import adopt_corpus_names

        adopt_corpus_names(settings, file, _corpus_root(settings_path))
    toks = [el for el in root.iter() if _local(el.tag) == "tok"]
    count = lambda name: sum(1 for el in root.iter() if _local(el.tag) == name)
    attributes = {}
    for internal in ANNOTATION_ATTRS:
        xml = _xml_name(settings, internal)
        n = sum(1 for t in toks if (t.get(xml) or "").strip() not in ("", "_"))
        if n:
            attributes[internal] = {"attribute": xml, "tokens": n}
    changes = []
    for el in root.iter():
        if _local(el.tag) == "change":
            text = " ".join(" ".join(t.strip() for t in el.itertext() if t.strip()).split())
            changes.append({k: v for k, v in (("when", el.get("when")), ("who", el.get("who")), ("tasks", el.get("tasks")), ("text", text[:200])) if v})

    # the run
    language_source = "given" if language else None
    if not language:
        language = _document_language(root)
        language_source = "document" if language else None
    if not language and settings is not None and settings.default_language:
        language, language_source = settings.default_language, "corpus"
    chosen = {"backend": backend, "model": model, "source": "given"} if (backend or model) else None
    if not chosen and language and settings is not None:
        pref = settings.get_flexipipe_preference(language)
        if pref:
            chosen = {"backend": pref.get("backend"), "model": pref.get("model"), "source": "corpus"}
    if not chosen and language:
        from ._cli import _auto_choice_for_language

        auto = _auto_choice_for_language(language)
        if auto:
            chosen = dict(auto, source="flexipipe")
    from ._cli import _default_tasks_for_backend

    run_tasks = {t.strip() for t in tasks.split(",") if t.strip()} if tasks else _default_tasks_for_backend((chosen or {}).get("backend"))
    run_tasks |= set(TASK_MANDATORY)
    # layers the corpus leaves out: their only step is not run, and they are not written
    excluded = set(getattr(settings, "excluded_layers", None) or ()) if settings is not None else set()
    for layer, task in (("pron", "pron"), ("reg", "normalize"), ("lemma", "lemmatize")):
        if layer in excluded:
            run_tasks.discard(task)
    keep_set = {k.strip().lower() for k in (keep or "").split(",") if k.strip()}
    overwrites, kept, new_attributes = [], [], []
    for task in sorted(run_tasks):
        for internal in TASK_ATTRS.get(task, []):
            if internal in excluded:
                continue
            if internal in attributes:
                entry = dict(attributes[internal], task=task, name=internal)
                (kept if internal in keep_set or "all" in keep_set else overwrites).append(entry)
            elif toks or "tokenize" in run_tasks:
                new_attributes.append({"name": internal, "attribute": _xml_name(settings, internal), "task": task})
    from .corpus_models import corpus_root as _root_of, in_use as _in_use

    _cm = _in_use(_root_of(settings_path), (getattr(settings, "training", None) or {}).get("use")) if settings_path else None
    _corpus_model = str(_cm) if _cm else None
    # who does which step (task_routing): the main backend, a backend given for the step, the corpus's
    from .task_routing import parse_routes, plan, SECONDARY

    routes, missing = plan(
        run_tasks, (chosen or {}).get("backend"), (chosen or {}).get("model"),
        parse_routes(task_backend or []), (getattr(settings, "task_backends", None) or {}) if settings else {},
        language=language,
        corpus_model=_corpus_model,
    )
    # who would do each step that can be chosen (for the steps not asked for too: a page can offer only those)
    all_routes, nobody = plan(
        ("tokenize", "segment", "normalize", "lemmatize", "tag", "parse", "ner", "pron"),
        (chosen or {}).get("backend"), (chosen or {}).get("model"),
        parse_routes(task_backend or []), (getattr(settings, "task_backends", None) or {}) if settings else {},
        language=language,
        corpus_model=_corpus_model,
    )
    providers = {t: {"backend": r.backend, "model": r.model, "source": r.source} for t, r in all_routes.items()}
    providers.update({t: None for t in nobody})
    steps = [{"task": t, "backend": r.backend, "model": r.model, "source": r.source} for t, r in sorted(routes.items())]
    steps += [{"task": t, "backend": None, "model": None, "source": None, "can": list(SECONDARY.get(t, ()))} for t in missing]
    # a model on disk (a pronunciation model, a flexitag model): where it comes from, in words
    from .model_card import describe, model_card

    corpus_root = Path(settings_path).parent.parent if settings_path else file.parent.parent
    cards: dict = {}

    def about(entry):
        model = (entry or {}).get("model")
        if model and "/" in model and Path(model).exists():
            if model not in cards:
                card = model_card(model, corpus_root)
                cards[model] = {"card": card, "about": describe(card)}
            entry.update(cards[model])
        return entry

    steps = [about(st) for st in steps]
    providers = {t: about(pr) for t, pr in providers.items()}
    if not tasks:  # the default steps: those nobody does are simply not among them
        run_tasks -= set(missing)
        steps = [st for st in steps if st["task"] not in missing]
        missing = []
    # a step nobody does is not run: it writes nothing
    overwrites = [o for o in overwrites if o["task"] not in missing]
    new_attributes = [a for a in new_attributes if a["task"] not in missing]
    adds = []
    if toks and not count("s") and "segment" in run_tasks:
        adds.append("sentences")
    if not toks and "tokenize" in run_tasks:
        adds.append("tokens")

    from .sentence_layout import SentenceLayout

    corpus_layout = (getattr(settings, "sentence_layout", None) if settings else None) or SentenceLayout()
    layout = corpus_layout.with_preset(sentence_layout) if sentence_layout else corpus_layout
    folder = file.parent
    return {
        "file": str(file),
        "tokens": len(toks),
        "dtoks": count("dtok"),
        "sentences": count("s"),
        "entities": count("name"),
        "attributes": attributes,
        "changes": changes,
        "settings": str(settings_path) if settings_path else None,
        "run": {
            "language": language,
            "language_source": language_source,  # given | document | corpus | None (detected when run)
            "backend": (chosen or {}).get("backend"),
            "model": (chosen or {}).get("model"),
            "model_source": (chosen or {}).get("source"),  # given | corpus | flexipipe | None
            "tasks": sorted(run_tasks),
            "steps": steps,  # per task: backend, model, source (main | given | corpus | flexipipe); None: nobody
            "missing": missing,
            "excluded": sorted(excluded),  # layers the corpus leaves out (never written)
            "providers": providers,  # every step that can be chosen: who would do it (None: nobody)
            "overwrites": overwrites,
            "kept": kept,  # existing values the run leaves as they are (--keep)
            "new_attributes": new_attributes,  # attributes the run writes that no token has yet
            "adds": adds,
            # how inserted sentences relate to the document's elements (sentence_layout.py)
            "sentence_layout": {"preset": layout.preset, "roles": layout.describe(), "corpus_preset": corpus_layout.preset},
        },
        "writable": os.access(file, os.W_OK) or os.access(folder, os.W_OK),
        "writable_detail": {"file": os.access(file, os.W_OK), "folder": os.access(folder, os.W_OK)},
    }
