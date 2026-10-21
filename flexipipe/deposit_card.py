"""The deposit form's fields for a trained model, from what the model folder knows about itself.

`deposit_fields(folder)` fills what can be filled — the backend, language, what it does, the held-out
scores, the corpus it was trained on, the training data's licence when the training files name one
unambiguously — for flexipipe-hub's deposit form (its field names). The person checks and completes them;
nothing here is personal (no names, no e-mail). A corpus model (``Resources/flexipipe-models/<x>/``) also
gives its corpus: ``corpus_of`` and, when the corpus has one, its tagset description (``Resources/tagset.xml``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

# what a model learned (train_report.LEARNED labels) → the hub's steps
_STEP_OF = {"UD part of speech": "tag", "tags": "tag", "morphological features": "tag",
            "lemmas": "lemmatize", "dependencies": "parse"}
_SCORE_NAMES = (("upos", "UPOS"), ("xpos", "tags"), ("feats", "features"), ("lemma", "lemmas"), ("las", "LAS"))

# Creative Commons licences as the TEI headers name them (a URL or the full name) → the hub's keys
_CC = re.compile(r"creativecommons\.org/(?:licenses|publicdomain)/([a-z-]+)/(\d\.\d)", re.I)
_CC_NAMES = {"by": "CC-BY", "by-sa": "CC-BY-SA", "by-nc": "CC-BY-NC", "by-nc-sa": "CC-BY-NC-SA",
             "by-nd": "CC-BY-ND", "by-nc-nd": "CC-BY-NC-ND", "zero": "CC0"}
_CC_WORDS = (("attribution-noncommercial-sharealike", "CC-BY-NC-SA"), ("attribution-noncommercial-noderivatives", "CC-BY-NC-ND"),
             ("attribution-noncommercial", "CC-BY-NC"), ("attribution-sharealike", "CC-BY-SA"),
             ("attribution-noderivatives", "CC-BY-ND"), ("attribution", "CC-BY"))


def corpus_of(folder) -> Optional[Path]:
    """The TEITOK project a corpus model belongs to (…/Resources/flexipipe-models/<x>/), else None."""
    folder = Path(folder).resolve()
    if folder.parent.name == "flexipipe-models" and folder.parent.parent.name == "Resources":
        return folder.parent.parent.parent
    return None


def corpus_tagset(folder) -> Optional[Path]:
    root = corpus_of(folder)
    if root is None:
        return None
    f = root / "Resources" / "tagset.xml"
    return f if f.is_file() else None


def licence_key(statement: str) -> Optional[str]:
    """A licence statement as a hub key (CC-BY-NC-SA-4.0 …), when it is a Creative Commons 4.0 one."""
    m = _CC.search(statement or "")
    if m:
        name = _CC_NAMES.get(m.group(1).lower())
        if name == "CC0":
            return "CC0-1.0"
        return f"{name}-{m.group(2)}" if name and m.group(2) == "4.0" else None
    words = re.sub(r"[\s_]+", "-", (statement or "").lower()).replace("non-commercial", "noncommercial").replace("share-alike", "sharealike")
    if "creative-commons" in words and "4.0" in words:
        for w, key in _CC_WORDS:
            if w in words:
                return f"{key}-4.0"
    return None


def _data_licence(summary: Optional[dict]) -> Optional[str]:
    """The training data's licence, when every file read names the same one (else: for the person)."""
    if not summary or summary.get("availability", {}).get("restricted"):
        return None
    licences = summary.get("licences") or {}
    if len(licences) != 1 or summary.get("with_licence") != summary.get("files"):
        return None
    return licence_key(next(iter(licences)))


def corpus_title(root: Optional[Path]) -> Optional[str]:
    """The corpus's display title (settings.xml <title display="…">), else None."""
    if root is None:
        return None
    try:
        import xml.etree.ElementTree as ET

        for el in ET.parse(root / "Resources" / "settings.xml").getroot().iter("title"):
            if el.get("display"):
                return el.get("display").strip()
    except Exception:
        return None
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def deposit_fields(folder, *, tagset_included: bool = False) -> Dict[str, object]:
    from .model_card import _manifest, model_card
    from .model_pack import public_name

    folder = Path(folder)
    root = corpus_of(folder)
    card = model_card(folder, root)
    manifest = _manifest(folder)
    corpus = (corpus_title(root) or root.name) if root else public_name(str(manifest.get("trained_on") or card.get("trained_on") or ""))
    backend = card.get("backend") or manifest.get("backend") or "flexitag"
    lang = card.get("language") or manifest.get("language") or ""

    steps: List[str] = []
    for label in card.get("learns") or []:
        step = _STEP_OF.get(label)
        if step and step not in steps:
            steps.append(step)
    if not steps and backend == "flexitag":
        steps = ["tag", "lemmatize"]
    pron = card.get("pron")
    if pron:
        steps.append("pron")

    what = ", ".join(card.get("learns") or []) or "tags and lemmas"
    description = f"A {backend} model trained on the {corpus} corpus: {what}"
    if pron:
        description += f"; pronunciations (@{pron.get('attribute')}" + (", with the next word" if pron.get("context") else "") + ")"
    if card.get("nlpform") == "reg":
        description += "; it tags the normalized forms"
    description += "."

    scores = card.get("scores") or {}
    shown = [f"{label} {100 * scores[k]:.1f}%" for k, label in _SCORE_NAMES if isinstance(scores.get(k), (int, float))]
    ps = (pron or {}).get("scores") or {}
    if isinstance(ps.get("all"), (int, float)):
        shown.append(f"pronunciation {100 * ps['all']:.1f}%" + (f" (new words {100 * ps['unknown']:.1f}%)" if isinstance(ps.get("unknown"), (int, float)) else ""))
    evaluation = ""
    if shown:
        n = scores.get("sentences")
        evaluation = f"On documents of {corpus} held out from the training" + (f" ({n} sentences)" if n else "") + ": " + ", ".join(shown) + "."

    fields: Dict[str, object] = {
        "backend": backend,
        "model": _slug(f"{lang}-{corpus}" if lang else corpus),
        "title": corpus,
        "description": description,
        "language_iso": lang,
        "steps": steps,
        "evaluation": evaluation,
        "data_name": corpus,
    }
    key = _data_licence(manifest.get("corpus_licence"))
    if key:
        fields["data_licence"] = key
    if tagset_included:
        fields["tagset"] = f"the {corpus} corpus's own tagset (described in tagset.xml, in the package)"
    return {k: v for k, v in fields.items() if v}
