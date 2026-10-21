"""A corpus as a release package for UD (a treebank) or UDMorph (UD annotation without dependency relations).

    corpus  -> documents as CoNLL-U (TEITOK files through flexipipe's own export, CoNLL-U files as they are)
            -> what they annotate (lemma, UPOS, XPOS, features, dependencies): UD when the trees are complete,
               UDMorph otherwise
            -> train / dev / test: UD's sizes (UDMorph: 80/10/10), on document boundaries where documents are small
               enough, else in runs of consecutive sentences; stable: sentences of an earlier release keep their split
            -> the repository's files (README and LICENSE: release_readme.py) and a report

UD: https://universaldependencies.org/contributing/repository_files.html (files, splits, README metadata; push to dev).
UDMorph: the UDMorph organisation on GitHub (repositories ``xxx-variant-name`` with ``NAME-train/dev/test.conllu``).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

SPLITS = ("train", "dev", "test")
FIELDS = ("lemma", "upos", "xpos", "feats", "deps")  # deps: HEAD and DEPREL


@dataclass
class Sent:
    sent_id: str
    lines: List[str]          # the sentence's lines, comments included
    words: int


@dataclass
class Doc:
    id: str
    head: List[str]           # the document's own comment lines (# newdoc id = …, its metadata)
    sents: List[Sent] = field(default_factory=list)
    source: str = ""          # the file it comes from

    @property
    def words(self) -> int:
        return sum(s.words for s in self.sents)


# ---- reading

def _files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            sub = p / "xmlfiles" if (p / "xmlfiles").is_dir() else p  # a TEITOK project: its documents
            for f in sorted(sub.rglob("*")):
                if f.suffix in (".conllu", ".xml") and f.is_file():
                    yield f
        else:
            yield p


def _conllu_of(path: Path, xpos_attr: Optional[str], sentence_comments: Optional[dict]) -> str:
    if path.suffix == ".conllu":
        return path.read_text(encoding="utf-8", errors="replace")
    from .conllu import document_to_conllu
    from .teitok import load_teitok

    doc = load_teitok(str(path), xpos_attr=xpos_attr)
    if sentence_comments:
        doc.meta["_sentence_comments"] = sentence_comments
    return document_to_conllu(doc, generator="")


def parse(text: str, fallback_id: str) -> List[Doc]:
    """CoNLL-U text as documents (# newdoc), each with its sentences."""
    docs: List[Doc] = []
    cur: Optional[Doc] = None
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip("\n").split("\n")
        if not any(l and not l.startswith("#") for l in lines):
            continue
        head, rest = [], lines
        for i, l in enumerate(lines):
            if l.startswith("# newdoc"):
                m = re.match(r"# newdoc(?: id = (.*))?", l)
                cur = Doc(id=(m.group(1) or "").strip() or f"{fallback_id}-{len(docs) + 1}", head=[])
                docs.append(cur)
                # the document's comments: from # newdoc up to the sentence's own (# sent_id / # text)
                j = i
                while j < len(lines) and lines[j].startswith("#") and not re.match(r"# (sent_id|text) =", lines[j]):
                    cur.head.append(lines[j])
                    j += 1
                rest = lines[:i] + lines[j:]
                break
        if cur is None:
            cur = Doc(id=fallback_id, head=[f"# newdoc id = {fallback_id}"])
            docs.append(cur)
        sid = next((l.split("=", 1)[1].strip() for l in rest if l.startswith("# sent_id")), "")
        words = sum(1 for l in rest if l and not l.startswith("#") and re.match(r"^\d+\t", l))
        cur.sents.append(Sent(sent_id=sid, lines=rest, words=words))
    return docs


def load(paths: Iterable[Path], *, xpos_attr: Optional[str] = "xpos", sentence_comments: Optional[dict] = None) -> List[Doc]:
    docs: List[Doc] = []
    for f in _files(paths):
        for d in parse(_conllu_of(f, xpos_attr, sentence_comments), f.stem):
            d.source = f.name
            docs.append(d)
    _unique_ids(docs)
    return docs


def _unique_ids(docs: List[Doc]) -> None:
    """sent_id: present and unique in the release (UD's rule): missing or repeated ones get the document's id."""
    seen: Counter = Counter(s.sent_id for d in docs for s in d.sents)
    used = set(seen)
    n = 0
    for d in docs:
        for s in d.sents:
            n += 1
            if s.sent_id and seen[s.sent_id] == 1:
                continue
            new = base = f"{d.id}-{s.sent_id or n}"
            k = 1
            while new in used:  # the same id twice in one document too
                k += 1
                new = f"{base}-{k}"
            used.add(new)
            s.lines = [l for l in s.lines if not l.startswith("# sent_id")]
            s.lines.insert(0, f"# sent_id = {new}")
            s.sent_id = new


# ---- what is annotated

def coverage(docs: List[Doc]) -> Dict[str, float]:
    """Per field the share of words that have it (deps: HEAD and DEPREL both)."""
    have: Counter = Counter()
    total = 0
    for d in docs:
        for s in d.sents:
            for l in s.lines:
                if not l or l.startswith("#") or not re.match(r"^\d+\t", l):
                    continue
                c = l.split("\t")
                if len(c) < 10:
                    continue
                total += 1
                for name, i in (("lemma", 2), ("upos", 3), ("xpos", 4), ("feats", 5)):
                    if c[i] != "_":
                        have[name] += 1
                if c[6] != "_" and c[7] != "_":
                    have["deps"] += 1
    return {f: (have[f] / total if total else 0.0) for f in FIELDS}


def _complete(d: Doc) -> bool:
    """Every word of the document has a HEAD and a DEPREL."""
    for s in d.sents:
        for l in s.lines:
            if l and re.match(r"^\d+\t", l):
                c = l.split("\t")
                if len(c) < 8 or c[6] == "_" or c[7] == "_":
                    return False
    return True


def complete_docs(docs: List[Doc]) -> Tuple[List[Doc], List[Doc]]:
    """UD: only documents with a complete tree for every sentence. Returns (kept, left out)."""
    kept, out = [], []
    for d in docs:
        (kept if _complete(d) else out).append(d)
    return kept, out


def choose_target(cov: Dict[str, float]) -> str:
    """UD needs complete trees; with UPOS but without (complete) dependencies the data belongs in UDMorph."""
    return "ud" if cov["deps"] >= 0.999 else "udmorph"


# ---- splits

def targets(words: int, target: str) -> Dict[str, int]:
    """The words each split should get. UD: under 20K all test; up to 110K 10K test and 10% of the rest dev; above
    10% test and 10% dev. UDMorph: 80/10/10 (as UDMorph's own distribute.py)."""
    if target == "udmorph":
        return {"train": words - 2 * (words // 10), "dev": words // 10, "test": words // 10}
    if words < 20000:
        return {"train": 0, "dev": 0, "test": words}
    if words <= 110000:
        test = 10000
        dev = (words - test) // 10
        return {"train": words - test - dev, "dev": dev, "test": test}
    return {"train": words - 2 * (words // 10), "dev": words // 10, "test": words // 10}


@dataclass
class Unit:
    """What goes into one split together: a document, or a run of its consecutive sentences."""
    id: str
    doc: Doc
    sents: List[Sent]

    @property
    def words(self) -> int:
        return sum(s.words for s in self.sents)


def units(docs: List[Doc], target_words: Dict[str, int]) -> List[Unit]:
    """Documents as units when they are small next to the smallest split; larger ones in runs of consecutive
    sentences (about a fifth of the smallest split), so that every split can get its share."""
    smallest = min((n for n in target_words.values() if n > 0), default=0)
    size = max(smallest // 5, 200) if smallest else 0
    out: List[Unit] = []
    for d in docs:
        if not size or d.words <= size:
            out.append(Unit(id=d.id, doc=d, sents=list(d.sents)))
            continue
        run: List[Sent] = []
        for s in d.sents:
            run.append(s)
            if sum(x.words for x in run) >= size:
                out.append(Unit(id=f"{d.id}#{run[0].sent_id}", doc=d, sents=run))
                run = []
        if run:
            out.append(Unit(id=f"{d.id}#{run[0].sent_id}", doc=d, sents=run))
    return out


def assign(us: List[Unit], target_words: Dict[str, int], previous: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """sent_id -> split. Sentences of an earlier release keep their split; the other units go, in corpus order, to
    the split furthest below its share (so every part of the corpus is spread over the splits, deterministically)."""
    previous = previous or {}
    out: Dict[str, str] = {}
    got: Counter = Counter()
    rest: List[Unit] = []
    for u in us:
        old = [previous[s.sent_id] for s in u.sents if s.sent_id in previous]
        if old:
            split = Counter(old).most_common(1)[0][0]
            for s in u.sents:  # a new sentence in an old unit goes with it; old ones keep theirs
                out[s.sent_id] = previous.get(s.sent_id, split)
                got[out[s.sent_id]] += s.words
        else:
            rest.append(u)
    total = sum(target_words.values()) or 1
    for u in sorted(rest, key=lambda u: _order(u.id)) if previous else rest:
        def deficit(sp: str) -> float:
            return (target_words[sp] - got[sp]) / total
        split = max((sp for sp in SPLITS if target_words[sp] > 0), key=deficit, default="test")
        for s in u.sents:
            out[s.sent_id] = split
        got[split] += u.words
    return out


def _order(uid: str) -> str:
    return hashlib.sha1(uid.encode("utf-8")).hexdigest()


def read_splits(path: Path) -> Dict[str, str]:
    """A splits file (sent_id TAB split), or the CoNLL-U files of an earlier release (a folder)."""
    out: Dict[str, str] = {}
    if path.is_dir():
        for f in path.glob("*.conllu"):
            m = re.search(r"-(train|dev|test)\.conllu$", f.name)
            if not m:
                continue
            for l in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if l.startswith("# sent_id"):
                    out[l.split("=", 1)[1].strip()] = m.group(1)
        return out
    for l in path.read_text(encoding="utf-8").splitlines():
        if "\t" in l:
            sid, sp = l.split("\t", 1)
            if sp.strip() in SPLITS:
                out[sid] = sp.strip()
    return out


# ---- writing

def strip_deps(line: str) -> str:
    """UDMorph: no dependency relations (HEAD, DEPREL, DEPS empty)."""
    if not line or line.startswith("#"):
        return line
    c = line.split("\t")
    if len(c) == 10:
        c[6] = c[7] = c[8] = "_"
    return "\t".join(c)


def split_texts(docs: List[Doc], split_of: Dict[str, str], *, target: str) -> Dict[str, Tuple[str, int, int]]:
    """Per split its CoNLL-U (documents in corpus order, # newdoc where a document starts in that split), sentences
    and words."""
    parts: Dict[str, List[str]] = {sp: [] for sp in SPLITS}
    n: Dict[str, Counter] = {sp: Counter() for sp in SPLITS}
    for d in docs:
        started = set()
        for s in d.sents:
            sp = split_of[s.sent_id]
            lines = list(s.lines)
            if sp not in started:
                lines = d.head + lines
                started.add(sp)
            if target == "udmorph":
                lines = [strip_deps(l) for l in lines]
            parts[sp].append("\n".join(lines))
            n[sp]["sents"] += 1
            n[sp]["words"] += s.words
    return {sp: ("\n\n".join(parts[sp]) + "\n\n", n[sp]["sents"], n[sp]["words"]) for sp in SPLITS if parts[sp]}
