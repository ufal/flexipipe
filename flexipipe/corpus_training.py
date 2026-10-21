"""What can be trained on a TEITOK corpus (`flexipipe info training`): the annotations its documents
have (under the corpus's own attribute names, from settings.xml), and for every trainable backend
(train_report.TRAINABLE) whether it is installed, what it would learn from this corpus, and what it
lacks. A sample of the documents is read (SAMPLE files) — enough to say what is there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

SAMPLE = 40
# the annotations a training can use, in the order a page lists them
ANNOTATIONS = ("xpos", "upos", "feats", "lemma", "head", "deprel", "reg", "expan", "trslit")
LABELS = {"xpos": "tags", "upos": "UD part of speech", "feats": "morphological features", "lemma": "lemmas",
          "head": "dependency heads", "deprel": "dependency relations", "reg": "normalized forms",
          "expan": "expansions", "trslit": "transliterations"}
# what each annotation teaches (a step of task_registry)
STEP_OF = {"xpos": "tag", "upos": "tag", "feats": "tag", "lemma": "lemmatize", "head": "parse", "deprel": "parse",
           "reg": "normalize", "expan": "normalize", "trslit": "normalize"}
# a transcription of the word (a pronunciation candidate): most values resemble the form, a fair
# share differs from it (not a copy, not an id or a reference)
# attributes named as a transcription are candidates whatever their values look like (IPA, SAMPA)
TRANSCRIPTION_NAMES = ("pron", "phon", "ipa", "sampa", "xsampa", "x-sampa", "phonetic", "phonemic", "transcription", "trans")
TRANSCRIPTION_SIMILAR = 0.6
TRANSCRIPTION_DIFFERENT = 0.05


def _similar(a: str, b: str) -> bool:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.5


def _local(tag) -> str:
    return tag.split("}", 1)[1] if isinstance(tag, str) and "}" in tag else str(tag)


def annotations(root: Path, settings=None, sample: int = SAMPLE) -> Dict[str, dict]:
    """{internal: {attribute, tokens}} for the annotations found on the <tok>s of a sample of the
    corpus's documents; also every other token attribute (``other``: candidates for a pronunciation)."""
    from lxml import etree

    from .teitok_settings import DEFAULT_ATTRIBUTE_MAPPINGS

    # every name an annotation may have here (the settings' mapping, flexipipe's defaults: reg / nform /
    # norm …); the one the corpus uses most is reported
    candidates = {}
    for a in ANNOTATIONS:
        names_a = list(settings.attribute_mappings.get(a, [])) if settings else []
        names_a += [n for n in DEFAULT_ATTRIBUTE_MAPPINGS.get(a, [a]) + [a] if n not in names_a]
        candidates[a] = names_a
    all_names = {n for names_a in candidates.values() for n in names_a}
    per_name = {a: {} for a in ANNOTATIONS}
    other: Dict[str, int] = {}
    resemble: Dict[str, list] = {}  # attribute -> [similar to the form, different from it]
    tokens = docs = 0
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)
    for path in sorted((root / "xmlfiles").rglob("*.xml"))[:sample]:
        try:
            tree = etree.parse(str(path), parser)
        except (OSError, etree.XMLSyntaxError):
            continue
        docs += 1
        for el in tree.iter():
            if _local(el.tag) not in ("tok", "dtok"):
                continue
            tokens += 1
            for a, names_a in candidates.items():
                for xml in names_a:
                    if (el.get(xml) or "").strip() not in ("", "_"):
                        per_name[a][xml] = per_name[a].get(xml, 0) + 1
                        break
            form = el.get("form") or "".join(el.itertext()).strip()
            for k, v in el.attrib.items():
                if k not in all_names and k not in ("id", "form", "ord", "join", "corresp", "sameAs", "tokid") and "}" not in k:
                    other[k] = other.get(k, 0) + 1
                    if form and v.strip() not in ("", "_", "--"):
                        r = resemble.setdefault(k, [0, 0])
                        r[0] += _similar(form, v)
                        r[1] += v.strip().lower() != form.lower()
    found = {}
    for a in ANNOTATIONS:
        if per_name[a]:
            name, n = max(per_name[a].items(), key=lambda kv: kv[1])
            found[a] = {"attribute": name, "tokens": sum(per_name[a].values()), "label": LABELS[a],
                        # every name in use, most used first (a training reads them all: --reg nform,norm)
                        "attributes": [k for k, _ in sorted(per_name[a].items(), key=lambda kv: -kv[1])]}
    named = lambda k: k.lower() in TRANSCRIPTION_NAMES or any(k.lower().startswith(n) for n in ("ipa", "sampa", "phon", "pron"))
    transcriptions = {k: n for k, n in sorted(other.items(), key=lambda kv: -kv[1])
                      if n and (named(k) or resemble.get(k, [0, 0])[0] >= TRANSCRIPTION_SIMILAR * n
                                and resemble[k][1] >= TRANSCRIPTION_DIFFERENT * n)}
    return {"documents": docs, "tokens": tokens, "found": found,
            "other": dict(sorted(other.items(), key=lambda kv: -kv[1])),
            "transcriptions": transcriptions}  # attributes that look like a pronunciation of the word


def training_options(root, settings=None) -> dict:
    """Per trainable backend: available, can (this corpus has what it needs), lacks, learns."""
    from ._cli import _required_annotations_for_backend
    from .backend_registry import get_backend_status
    from .train_report import TRAINABLE

    root = Path(root)
    ann = annotations(root, settings)
    have = set(ann["found"])
    backends = []
    for name, info in TRAINABLE.items():
        status = get_backend_status(name)
        need = _required_annotations_for_backend(name, ann["found"].get("xpos", {}).get("attribute"))
        lacks = [a for a in need if not set(a.split("|")) & have]
        learns = sorted({LABELS[a] for a in have if STEP_OF[a] in info["learns"] or a == "lemma" and "lemmatize" in info["learns"]})
        backends.append({
            "backend": name,
            "about": info["about"],
            "available": bool(status.get("available")),
            "install_hint": status.get("install_hint") or "",
            "needs": info["needs"],  # besides the data: language, a base model
            "lacks": [" or ".join(LABELS.get(x, x) for x in a.split("|")) for a in lacks],
            "can": bool(status.get("available")) and not lacks and "base model" not in info["needs"],
            "learns": learns,
        })
    # flexitag follows one tag sequence: the ones this corpus can give
    paths = [p for p, need in (("xpos", {"xpos"}), ("upos", {"upos"}), ("utot", {"upos", "feats"})) if need <= have]
    writes = {}
    if settings is not None:  # where each layer goes: the user's choice, else the automatic one (teitok_names)
        from .teitok_names import automatic_choices

        writes = automatic_choices(settings, root)
    return {"corpus": str(root), **ann, "backends": backends, "flexitag_paths": paths, "writes": writes}
