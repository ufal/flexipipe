"""A light grapheme-to-phoneme learner for flexitag's pronunciation attribute (forms not in the
vocabulary). Learned from (form, pronunciation) pairs of a corpus — e.g. ORTOFON's @phon.

- Alignment: each pair is aligned letter by letter (edit distance; identical and similar letters are
  cheap). A sound inserted between letters joins the letter before it (ě → je), a letter without a
  sound maps to the empty string, so every letter gets an output string.
- Learning: for every letter, the output in its context — every combination of 0 to MAX_CONTEXT
  letters on the left and on the right (one-sided contexts too), ``#`` at the word edges — counted; a
  context is kept only when its most frequent output differs from that of a context one letter
  shorter (IGTree-style pruning: the tables stay small).
- Prediction: per letter, the longest context seen often enough decides (a longer context needs more
  evidence: MIN_COUNT + EVIDENCE_PER_LETTER per letter of context); shorter ones are the back-off.
  Confidence: the mean share of the chosen outputs.

Strong for regular spellings (Czech); no dependency. Plan: dev/SPOKEN-PRONUNCIATION-AND-SENTENCES.md.

Context (optional, ``context="next"`` when training): how a word ends often depends on the next word
(Czech voicing assimilation: tak → tag before a voiced sound; m → ɱ before v, n → ŋ before k). The
next word's first letter (``#`` at a pause or the end) — language-neutral, no phonology table: the
vocabulary keeps, per level, the contexts that change the answer (seen at least CONTEXT_MIN times,
and ahead of the plain answer by CONTEXT_MARGIN in that context). Measured on ORTOFON (realised
@phon, flexitag's own tags): known forms 76.7% → 77.9% (words ending in an obstruent 59% → 67%).
The G2P does not use it: the next word's letter after the word end made unknown forms no better
(62.7% → 62.4%). For broad transcriptions it adds little: leave it off.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

MAX_CONTEXT = 3
MIN_COUNT = 2
# a longer context must be seen more often to decide: MIN_COUNT + EVIDENCE_PER_LETTER × its length
# (measured on ORTOFON: without it, long one-sided contexts with little evidence overfit)
EVIDENCE_PER_LETTER = 3
# context exceptions in the vocabulary: seen this often, and ahead of the plain answer by this share
CONTEXT_MIN = 5
CONTEXT_MARGIN = 0.1


def next_context(form: Optional[str]) -> str:
    """The context a following token gives: its first letter (lowercase); ``#`` for none, punctuation
    or anything else that is not a letter (a pause)."""
    if not form or not form[0].isalpha():
        return "#"
    return form[0].lower()


def _needed(left: int, right: int) -> int:
    return 1 if left + right == 0 else MIN_COUNT + EVIDENCE_PER_LETTER * (left + right)


def _base(ch: str) -> str:
    """The letter without diacritics (ř ~ r, ě ~ e): such pairs align cheaply."""
    return unicodedata.normalize("NFD", ch)[:1].lower()


def align(form: str, pron: str) -> Optional[List[str]]:
    """Per letter of `form`, its part of `pron` (concatenated: `pron`); None when they share too little."""
    n, m = len(form), len(pron)
    if not n:
        return None
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back: List[List[Tuple[int, int]]] = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if i < n and j < m:  # letter ↔ sound
                a, b = form[i].lower(), pron[j].lower()
                sub = 0.0 if a == b else 0.4 if _base(a) == _base(b) else 1.0
                if c + sub < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1], back[i + 1][j + 1] = c + sub, (i, j)
            if i < n and c + 0.8 < cost[i + 1][j]:  # a letter without a sound
                cost[i + 1][j], back[i + 1][j] = c + 0.8, (i, j)
            if j < m and c + 0.7 < cost[i][j + 1]:  # an extra sound
                cost[i][j + 1], back[i][j + 1] = c + 0.7, (i, j)
    if cost[n][m] > 0.6 * max(n, m) + 1:
        return None
    # walk back: letter i gets the sound(s) up to the next letter
    out = [""] * n
    i, j = n, m
    pending = ""
    while (i, j) != (0, 0):
        pi, pj = back[i][j]
        if pi == i - 1 and pj == j - 1:      # letter i-1 ↔ sound j-1
            out[i - 1] = pron[j - 1] + pending
            pending = ""
        elif pi == i - 1:                     # letter i-1 silent
            out[i - 1] = pending
            pending = ""
        else:                                 # extra sound j-1: joins the letter before it
            pending = pron[j - 1] + pending
        i, j = pi, pj
    if pending:  # sounds before the first letter
        out[0] = pending + out[0]
    return out


def _contexts(word: str, i: int) -> List[Tuple[int, int, str]]:
    """(left, right, key) for letter i: every combination of 0..MAX_CONTEXT letters on each side,
    longest first (one-sided contexts too: "z before a vowel" needs no left context)."""
    padded = "#" * MAX_CONTEXT + word + "#" * MAX_CONTEXT
    p = i + MAX_CONTEXT
    out = [(l, r, padded[p - l: p] + "[" + padded[p] + "]" + padded[p + 1: p + 1 + r])
           for l in range(MAX_CONTEXT + 1) for r in range(MAX_CONTEXT + 1)]
    return sorted(out, key=lambda x: (-(x[0] + x[1]), -x[1]))


def _shorter(key: str) -> List[str]:
    """The contexts one letter shorter on either side."""
    i, j = key.index("["), key.index("]")
    out = []
    if i > 0:
        out.append(key[1:])
    if j < len(key) - 1:
        out.append(key[:-1])
    return out


class LightG2P:
    def __init__(self, table: Optional[Dict[str, Dict[str, int]]] = None):
        self.table: Dict[str, Dict[str, int]] = table or {}

    @classmethod
    def train(cls, pairs: Iterable[Tuple[str, str, int]]) -> "LightG2P":
        """From (form, pronunciation, count) pairs."""
        counts: Dict[str, Counter] = defaultdict(Counter)
        for form, pron, n in pairs:
            form = form.lower()
            parts = align(form, pron)
            if parts is None:
                continue
            for i, out in enumerate(parts):
                for _, _, key in _contexts(form, i):
                    counts[key][out] += n
        # prune: keep a context only when its winner differs from that of a context one letter
        # shorter (on either side) — the rest adds nothing over the back-off
        table: Dict[str, Dict[str, int]] = {}
        for key, c in counts.items():
            if key.startswith("[") and key.endswith("]"):
                table[key] = dict(c.most_common(4))
                continue
            i, j = key.index("["), key.index("]")
            if sum(c.values()) < _needed(i, len(key) - j - 1):
                continue  # never decides: below its evidence threshold
            best = c.most_common(1)[0][0]
            parents = [counts[k] for k in _shorter(key) if k in counts]
            if not parents or any(p.most_common(1)[0][0] != best for p in parents):
                table[key] = dict(c.most_common(4))
        return cls(table)

    def predict(self, form: str) -> Tuple[str, float]:
        """The pronunciation of `form` and a confidence (0..1)."""
        word = form.lower()
        out, shares = [], []
        for i, ch in enumerate(word):
            chosen = None
            for left, right, key in _contexts(word, i):
                c = self.table.get(key)
                if c and sum(c.values()) >= _needed(left, right):
                    chosen = c
                    break
            if not chosen:
                out.append(ch)  # a letter never seen: as it is
                shares.append(0.0)
                continue
            best, n = max(chosen.items(), key=lambda kv: kv[1])
            out.append(best)
            shares.append(n / sum(chosen.values()))
        return "".join(out), (sum(shares) / len(shares) if shares else 0.0)

    def to_json(self) -> dict:
        return {"type": "light-g2p", "max_context": MAX_CONTEXT, "min_count": MIN_COUNT,
                "evidence_per_letter": EVIDENCE_PER_LETTER, "table": self.table}

    @classmethod
    def from_json(cls, data: dict) -> "LightG2P":
        return cls(data.get("table") or {})


# the vocabulary levels, most specific first; each is used only when its combination was seen in
# training (form + lemma + xpos separates same-tag homographs; upos + feats serves corpora whose
# xpos is coarse or missing)
LEVELS = ("form+lemma+xpos", "form+xpos", "form+upos+feats", "form+upos", "form")


def _feats(feats: str, keys: Optional[List[str]]) -> str:
    """The feats that count (all, or only `keys`, e.g. Case and Number), in a fixed order."""
    if not feats or feats == "_":
        return ""
    pairs = sorted(p for p in feats.split("|") if "=" in p)
    if keys:
        pairs = [p for p in pairs if p.split("=", 1)[0] in keys]
    return "|".join(pairs)


def _level_key(level: str, form: str, lemma: str, xpos: str, upos: str, feats: str) -> Optional[str]:
    """The lookup key of a token at a level; None when the token lacks what the level needs."""
    form = form.lower()
    parts = {"form": form, "lemma": lemma, "xpos": xpos, "upos": upos, "feats": feats}
    values = []
    for name in level.split("+"):
        v = parts[name]
        if name != "form" and (not v or v == "_"):
            return None
        values.append(v)
    return "\t".join(values)


class PronunciationModel:
    """flexitag's pronunciation attribute: a fallback chain over the vocabulary (LEVELS: form + lemma +
    xpos, form + xpos, form + upos + feats, form + upos, form) — the most specific combination seen
    in training decides — then the light G2P for forms not in the vocabulary. `predict` names the level
    that decided (``vocab:form+xpos`` … ``g2p``)."""

    def __init__(self, levels: Dict[str, Dict[str, str]], g2p: LightG2P, meta: Optional[dict] = None,
                 context: Optional[Dict[str, Dict[str, str]]] = None):
        self.levels, self.g2p, self.meta = levels, g2p, meta or {}
        self.feats_keys = (self.meta.get("feats") or None)
        # per level: "key<TAB>next" -> pronunciation, where the next word changes the answer
        self.context: Dict[str, Dict[str, str]] = context or {}

    @property
    def uses_context(self) -> bool:
        return self.meta.get("context") == "next"

    # the form level, for callers that ask whether a form is known
    @property
    def by_form(self) -> Dict[str, str]:
        return self.levels.get("form", {})

    @classmethod
    def train(cls, rows: Iterable, meta: Optional[dict] = None) -> "PronunciationModel":
        """From tokens: (form, xpos, pronunciation) or dicts with form, lemma, xpos, upos, feats, pron
        (and ``next``: next_context of the following token, used with meta context="next")."""
        meta = dict(meta or {})
        keys = meta.get("feats") or None
        use_context = meta.get("context") == "next"
        counts: Dict[str, Dict[str, Counter]] = {lv: defaultdict(Counter) for lv in LEVELS}
        in_context: Dict[str, Dict[str, Counter]] = {lv: defaultdict(Counter) for lv in LEVELS}
        for row in rows:
            following = "#"
            if isinstance(row, dict):
                form, lemma, xpos, upos, feats, pron = (row.get(k, "") or "" for k in ("form", "lemma", "xpos", "upos", "feats", "pron"))
                following = row.get("next") or "#"
            else:
                form, xpos, pron = row
                lemma = upos = feats = ""
            if not pron or pron in ("_", "--"):
                continue
            feats = _feats(feats, keys)
            for lv in LEVELS:
                key = _level_key(lv, form, lemma, xpos, upos, feats)
                if key is not None:
                    counts[lv][key][pron] += 1
                    if use_context:
                        in_context[lv][key + "\t" + following][pron] += 1
        g2p = LightG2P.train((k, c.most_common(1)[0][0], sum(c.values())) for k, c in counts["form"].items())
        levels = {lv: {k: c.most_common(1)[0][0] for k, c in counts[lv].items()} for lv in LEVELS if counts[lv]}
        context: Dict[str, Dict[str, str]] = {}
        for lv, table in in_context.items():
            for ck, c in table.items():
                total = sum(c.values())
                if total < CONTEXT_MIN:
                    continue
                plain = levels[lv][ck.rsplit("\t", 1)[0]]
                best, n = c.most_common(1)[0]
                if best != plain and (n - c.get(plain, 0)) / total > CONTEXT_MARGIN:
                    context.setdefault(lv, {})[ck] = best
        meta["levels"] = {lv: len(v) for lv, v in levels.items()}
        if use_context:
            meta["context_entries"] = {lv: len(v) for lv, v in context.items()}
        return cls(levels, g2p, meta, context)

    def lookup(self, form: str, xpos: str = "", lemma: str = "", upos: str = "", feats: str = "",
               following: str = "#") -> Optional[Tuple[str, str, float]]:
        """The vocabulary chain only: (pronunciation, ``vocab:<level>``, confidence), None for a form
        not in the vocabulary (for a G2P engine of choice: pron_backends). `following`: next_context
        of the next token (models trained with context="next"; the source then ends in ``+next``
        when the next word changed the answer)."""
        f = _feats(feats, self.feats_keys)
        for lv in LEVELS:
            table = self.levels.get(lv)
            if not table:
                continue
            key = _level_key(lv, form, lemma, xpos, upos, f)
            if key is not None and key in table:
                conf = 1.0 if lv != "form" else 0.9
                in_context = self.context.get(lv, {}).get(key + "\t" + (following or "#"))
                if in_context is not None:
                    return in_context, f"vocab:{lv}+next", conf
                return table[key], f"vocab:{lv}", conf
        return None

    def predict(self, form: str, xpos: str = "", lemma: str = "", upos: str = "", feats: str = "",
                following: str = "#") -> Tuple[str, str, float]:
        """(pronunciation, source, confidence): source ``vocab:<level>`` or ``g2p``."""
        found = self.lookup(form, xpos, lemma, upos, feats, following)
        if found:
            return found
        pron, conf = self.g2p.predict(form)
        return pron, "g2p", conf

    def save(self, path) -> None:
        data = {"meta": self.meta, "levels": self.levels, "g2p": self.g2p.to_json()}
        if self.context:
            data["context"] = self.context
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "PronunciationModel":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        levels = data.get("levels")
        if levels is None:  # the first format: (form, tag) and form only
            levels = {"form+xpos": data.get("by_form_tag", {}), "form": data.get("by_form", {})}
        return cls(levels, LightG2P.from_json(data["g2p"]), data.get("meta"), data.get("context"))
