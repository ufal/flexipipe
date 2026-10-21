"""What a model folder is, in words a page can show: where it came from (trained here, on which corpus
and when; or installed), and for a pronunciation model what it does (vocabulary, next-word context,
G2P for other words). Reads only the head of the model files (their metadata comes first), so a
status page does not load a model of tens of megabytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_HEADS = (65536, 1 << 20, 16 << 20)  # read more only when the metadata is that large (big tagsets)


def _head_object(path: Path, key: str) -> dict:
    """The JSON object under `key` at the start of a JSON file ({"key": {...}, …}), or {}."""
    for size in _HEADS:
        try:
            with open(path, encoding="utf-8") as fh:
                head = fh.read(size)
            start = head.index('"' + key + '"')
            start = head.index("{", start)
            value, _ = json.JSONDecoder().raw_decode(head[start:])
            return value if isinstance(value, dict) else {}
        except OSError:
            return {}
        except ValueError:
            if len(head) < size:  # the whole file was read: it is not there
                return {}
    return {}


def _manifest(folder: Path) -> dict:
    try:
        data = json.loads((folder / "flexipipe-model.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _where_trained(card: dict, source, corpus_root) -> None:
    """locally_trained / trained_on / this_corpus from the training data's folder (it exists here:
    trained on this machine)."""
    source = Path(source) if source else None
    if source is None or not source.exists():
        card["locally_trained"] = False
        return
    if source.is_file():
        source = source.parent
    corpus = source.parent if source.name == "xmlfiles" else source
    card["trained_on"] = corpus.name
    card["locally_trained"] = True
    if corpus_root is not None:
        try:
            card["this_corpus"] = corpus.resolve() == Path(corpus_root).resolve()
        except OSError:
            card["this_corpus"] = False


def model_card(path, corpus_root: Optional[Path] = None) -> dict:
    """Facts about a flexitag model folder (or a file in it): name, created, trained_on (a corpus
    folder on this machine: locally trained), this_corpus, language; pron (attribute, tokens,
    context, languages) when it has a pronunciation model."""
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    card: dict = {"name": folder.name, "path": str(folder)}
    manifest = _manifest(folder)
    meta = _head_object(folder / "model_vocab.json", "metadata")
    if manifest:  # written by training (train_report): any backend
        card["backend"] = manifest.get("backend")
        card["created"] = (manifest.get("created") or "")[:10] or None
        card["language"] = manifest.get("language")
        card["learns"] = manifest.get("learns") or []
        card["options"] = manifest.get("options") or {}
        card["nlpform"] = card["options"].get("nlpform")
        if manifest.get("scores"):
            card["scores"] = manifest["scores"]  # upos, xpos, feats, lemma, uas, las on the held-out set
            card["held_out"] = manifest.get("held_out")
        _where_trained(card, manifest.get("trained_on"), corpus_root)
    elif meta:
        card["created"] = (meta.get("creation_date") or "")[:10] or None
        card["language"] = meta.get("language")
        card["tag_attribute"] = meta.get("tag_attribute")
        metrics = meta.get("final_metrics") or {}
        if metrics.get("pos_accuracy") is not None:
            card["scores"] = {"tags": metrics.get("pos_accuracy"), "lemmas": metrics.get("lemma_accuracy")}
        card["backend"] = "flexitag"
        card["nlpform"] = meta.get("nlpform")
        sources = meta.get("source_folders") or []
        _where_trained(card, sources[0] if sources else None, corpus_root)
    pron = _head_object(folder / "model_pron.json", "meta")
    if pron:
        card["pron"] = {
            "attribute": pron.get("attribute"),
            "tokens": pron.get("tokens"),
            "context": pron.get("context") == "next",
            "language": pron.get("language"),
            "phonetisaurus": (folder / "model_pron.fst").exists(),
            "scores": pron.get("scores"),  # held-out: all / known / unknown forms
        }
    return card


def model_nlpform(model) -> Optional[str]:
    """"reg" when the model (a folder or a file in one) was trained on the normalized forms, else None."""
    p = Path(str(model))
    if not p.exists():
        return None
    folder = p if p.is_dir() else p.parent
    value = (_manifest(folder).get("options") or {}).get("nlpform") or _head_object(folder / "model_vocab.json", "metadata").get("nlpform")
    return value if value and value != "form" else None


def describe(card: dict) -> str:
    """One line for a page or the terminal: "locally trained on this corpus (ortofon), 2026-10-18"."""
    if card.get("locally_trained"):
        where = "this corpus" if card.get("this_corpus") else f"the corpus {card.get('trained_on')}"
        text = f"locally trained on {where}" + (f" ({card['trained_on']})" if card.get("this_corpus") else "")
    else:
        text = f"the installed model {card.get('name')}"
    if card.get("created"):
        text += f", {card['created']}"
    if card.get("nlpform") == "reg":
        text += "; tags the normalized forms"
    pron = card.get("pron")
    if pron:
        text += "; known words from its vocabulary" + (" (the next word taken into account)" if pron.get("context") else "")
        text += ", other words by " + ("Phonetisaurus" if pron.get("phonetisaurus") else "rule (light G2P)")
    return text
