"""A language's UD inventory drafted from a corpus: the relations, UPOS and features (with their values per UPOS) the
corpus uses, optionally on top of a related language's UD data. For languages UD has no data for yet.

The file is the one TEITOK's tree editor (teitok-extras, deptree-edit) reads as ``Resources/deptree-edit-ud-LANG.json``
— the short form of UD's validator data (UniversalDependencies/tools, data/feats.json and data/deprels.json):

    {"lang": "xx", "upos": [...], "deprels": [...],
     "feats": {"Number": {"values": ["Sing", "Plur"], "byupos": {"NOUN": ["Sing", "Plur"]}}},
     "draft": {...}}

``draft`` is what it was made from and the evidence: the counts per relation, UPOS and feature value per UPOS, and
what UD does not know (``new``: to document, or to correct), and what the base language does not permit
(``beyond_base``: "ADJ Degree=Pos"). TEITOK's language-data page completes the draft.
"""
from __future__ import annotations

import datetime
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

UD_DATA_URL = "https://raw.githubusercontent.com/UniversalDependencies/tools/master/data"


# ---- reading corpora

def _files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix in (".conllu", ".xml") and f.is_file():
                    yield f
        else:
            yield p


def _documents(paths: Iterable[Path], xpos_attr: Optional[str] = None):
    from .conllu import conllu_to_document
    from .teitok import load_teitok

    for f in _files(paths):
        if f.suffix == ".conllu":
            yield f, conllu_to_document(f.read_text(encoding="utf-8", errors="replace"), doc_id=f.stem)
        else:
            yield f, load_teitok(str(f), xpos_attr=xpos_attr)


def _words(doc):
    """The syntactic words: the words of a multiword token instead of the token."""
    for s in doc.sentences:
        for t in s.tokens:
            if t.is_mwt and t.subtokens:
                yield from t.subtokens
            else:
                yield t


def _feats(s: str) -> List[tuple]:
    out = []
    for fv in (s or "").split("|"):
        if "=" in fv:
            k, v = fv.split("=", 1)
            for one in v.split(","):  # Gender=Fem,Masc: two values of the feature
                out.append((k, one))
    return out


def count(paths: Iterable[Path], xpos_attr: Optional[str] = None) -> dict:
    """What the corpus uses: relations, UPOS, feature values per UPOS, and the files read."""
    rels: Counter = Counter()
    upos: Counter = Counter()
    feats: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    files, words = [], 0
    for f, doc in _documents(paths, xpos_attr):
        files.append(str(f))
        for w in _words(doc):
            words += 1
            if w.deprel and w.deprel != "_":
                rels[w.deprel] += 1
            if w.upos and w.upos != "_":
                upos[w.upos] += 1
            for k, v in _feats(w.feats):
                feats[k][w.upos or "_"][v] += 1
    return {"files": files, "words": words, "deprels": dict(rels), "upos": dict(upos),
            "feats": {k: {u: dict(c) for u, c in by.items()} for k, by in feats.items()}}


# ---- UD's data

def load_ud_data(cache: Optional[Path] = None) -> tuple:
    """UD's feats.json and deprels.json: from the cache folder, else fetched (and kept there)."""
    import urllib.request

    out = []
    for name in ("feats", "deprels"):
        f = cache / f"{name}.json" if cache else None
        if f and f.exists():
            out.append(json.loads(f.read_text(encoding="utf-8")))
            continue
        with urllib.request.urlopen(f"{UD_DATA_URL}/{name}.json", timeout=60) as r:
            text = r.read().decode("utf-8")
        if f:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text, encoding="utf-8")
        out.append(json.loads(text))
    return out[0], out[1]


def universal(feats_json: dict, deprels_json: dict) -> dict:
    """UD's universal relations and features with their universal values (from any language's entry)."""
    rels, fv = set(), defaultdict(set)
    for lang in deprels_json.get("deprels", {}).values():
        for rel, d in lang.items():
            if d.get("type") == "universal":
                rels.add(rel)
    for lang in feats_json.get("features", {}).values():
        for k, d in lang.items():
            if d.get("type") == "universal":
                fv[k].update(d.get("uvalues", []))
    return {"deprels": rels, "feats": fv}


def compact(feats_json: dict, deprels_json: dict, lang: str) -> Optional[dict]:
    """One language's UD data in short: permitted relations and features (values, values per UPOS)."""
    fl = feats_json.get("features", {}).get(lang)
    dl = deprels_json.get("deprels", {}).get(lang)
    if not fl and not dl:
        return None
    out = {"lang": lang, "deprels": sorted(r for r, d in (dl or {}).items() if d.get("permitted")), "feats": {}}
    for k, d in (fl or {}).items():
        if not d.get("permitted"):
            continue
        values = list(dict.fromkeys(d.get("uvalues", []) + d.get("lvalues", [])))
        byupos = {u: [v for v, ok in vs.items() if ok] for u, vs in d.get("byupos", {}).items()}
        out["feats"][k] = {"values": values, "byupos": {u: vs for u, vs in byupos.items() if vs}}
    return out


# ---- the draft

def draft(counts: dict, lang: str, *, base: Optional[dict] = None, ud: Optional[dict] = None,
          minimum: int = 1) -> dict:
    """The language file: what the corpus uses (at least ``minimum`` times), on top of ``base`` (a related language's
    compact UD data) when given. ``ud``: UD's universal sets, to say what is new to UD."""
    rels = {r for r, n in counts["deprels"].items() if n >= minimum}
    feats: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for k, by in counts["feats"].items():
        for u, vals in by.items():
            for v, n in vals.items():
                if n >= minimum:
                    feats[k][u].add(v)
    if base:
        rels |= set(base.get("deprels", []))
        for k, d in base.get("feats", {}).items():
            for u, vs in d.get("byupos", {}).items():
                feats[k][u].update(vs)
            feats[k]  # (touched: a feature of the base without values per UPOS is kept)
    out_feats = {}
    for k in sorted(feats):
        by = {u: sorted(vs) for u, vs in sorted(feats[k].items()) if u != "_" and vs}
        values = sorted({v for vs in feats[k].values() for v in vs} | set((base or {}).get("feats", {}).get(k, {}).get("values", [])))
        out_feats[k] = {"values": values, "byupos": by}
    new = {"deprels": [], "feats": []}
    if ud:
        base_rels = set((base or {}).get("deprels", []))
        new["deprels"] = sorted(r for r in rels if r not in ud["deprels"] and r not in base_rels)
        base_feats = (base or {}).get("feats", {})
        for k, d in out_feats.items():
            for v in d["values"]:
                if v not in ud["feats"].get(k, set()) and v not in base_feats.get(k, {}).get("values", []):
                    new["feats"].append(f"{k}={v}")
    # what the corpus uses that the base does not permit (for that UPOS): to decide on, one by one
    beyond = {"deprels": [], "feats": []}
    if base:
        beyond["deprels"] = sorted(r for r, n in counts["deprels"].items() if n >= minimum and r not in base.get("deprels", []))
        bf = base.get("feats", {})
        for k, by in sorted(counts["feats"].items()):
            for u, vals in sorted(by.items()):
                for v, n in sorted(vals.items()):
                    if n >= minimum and v not in bf.get(k, {}).get("byupos", {}).get(u, []):
                        beyond["feats"].append(f"{u} {k}={v}")
        beyond["feats"].sort()
    upos = sorted(u for u, n in counts["upos"].items() if n >= minimum)
    return {
        "lang": lang,
        "upos": upos,
        "deprels": sorted(rels),
        "feats": out_feats,
        "draft": {
            "made": datetime.date.today().isoformat(),
            "by": "flexipipe udlang",
            "files": len(counts["files"]),
            "words": counts["words"],
            "base": (base or {}).get("lang"),
            "minimum": minimum,
            "counts": {"deprels": counts["deprels"], "upos": counts["upos"], "feats": counts["feats"]},
            "new": new,
            "beyond_base": beyond,
        },
    }
