from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .doc import Document, Token


def _normalize_feats(feats: str) -> str:
    """Normalize a UD FEATS string (sorted, deduplicated)."""
    if not feats or feats == "_":
        return "_"
    parts = []
    for feat in feats.split("|"):
        feat = feat.strip()
        if not feat or "=" not in feat:
            continue
        k, v = feat.split("=", 1)
        parts.append((k.strip(), v.strip()))
    if not parts:
        return "_"
    parts.sort()
    return "|".join(f"{k}={v}" for k, v in parts)


def _feats_to_dict(feats: str) -> Dict[str, str]:
    if not feats or feats == "_":
        return {}
    result: Dict[str, str] = {}
    for feat in feats.split("|"):
        if "=" not in feat:
            continue
        key, value = feat.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _iter_vocab_entries(vocab: Dict) -> Iterable[Tuple[Dict, int]]:
    """Yield (analysis, count) tuples from a vocab dict."""
    for value in vocab.values():
        if isinstance(value, list):
            for analysis in value:
                if isinstance(analysis, dict):
                    yield analysis, int(analysis.get("count", 1))
        elif isinstance(value, dict):
            yield value, int(value.get("count", 1))


@dataclass
class TagMapping:
    """Bi-directional mapping between XPOS tags and UPOS/FEATS combinations."""

    xpos_to_combo: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    combo_to_xpos: Dict[Tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    xpos_lower_index: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_vocab(cls, vocab: Dict) -> "TagMapping":
        mapping = cls()
        mapping._ingest_vocab(vocab)
        return mapping

    @classmethod
    def from_model_file(cls, path: Path | str) -> "TagMapping":
        path = Path(path)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        vocab = data.get("vocab", data)
        return cls.from_vocab(vocab)

    def _ingest_vocab(self, vocab: Dict) -> None:
        for analysis, raw_count in _iter_vocab_entries(vocab):
            count = max(raw_count, 1)
            xpos = analysis.get("xpos") or ""
            upos = (analysis.get("upos") or "").strip()
            feats = _normalize_feats(analysis.get("feats", ""))

            if xpos:
                self.xpos_lower_index.setdefault(xpos.lower(), xpos)

            if xpos and upos:
                self.xpos_to_combo[xpos][(upos, feats)] += count
                self.combo_to_xpos[(upos, feats)][xpos] += count
            elif xpos and feats != "_":
                self.xpos_to_combo[xpos][("", feats)] += count
            elif upos:
                self.combo_to_xpos[(upos, feats)][xpos or ""] += count

    def guess_upos_feats(
        self,
        xpos: str,
        *,
        allow_partial: bool = True,
    ) -> Optional[Tuple[str, str]]:
        counter = self._counter_for_xpos(xpos, allow_partial=allow_partial)
        if not counter:
            return None
        (upos, feats), _ = counter.most_common(1)[0]
        normalized_feats = feats if feats and feats != "_" else "_"
        return (upos or "_", normalized_feats)

    def guess_xpos(
        self,
        upos: str,
        feats: str,
        *,
        allow_partial: bool = True,
    ) -> Optional[str]:
        upos = upos or "_"
        feats_norm = _normalize_feats(feats)
        counter = self.combo_to_xpos.get((upos, feats_norm))
        if counter:
            return counter.most_common(1)[0][0] or ""

        if not allow_partial:
            return None

        target_feats = _feats_to_dict(feats_norm)
        best_choice: Optional[Tuple[int, str]] = None

        for (candidate_upos, candidate_feats), xpos_counter in self.combo_to_xpos.items():
            if candidate_upos != upos:
                continue

            cand_feats_dict = _feats_to_dict(candidate_feats)
            if target_feats and not self._subset_of(target_feats, cand_feats_dict):
                continue

            xpos, count = xpos_counter.most_common(1)[0]
            if best_choice is None or count > best_choice[0]:
                best_choice = (count, xpos)

        return best_choice[1] if best_choice else None

    def enrich_token(
        self,
        token: Token,
        *,
        fill_xpos: bool = True,
        fill_upos: bool = True,
        fill_feats: bool = True,
        allow_partial: bool = True,
    ) -> bool:
        changed = False

        has_xpos = bool(token.xpos and token.xpos != "_")
        needs_upos = not token.upos or token.upos == "_"
        needs_feats = not token.feats or token.feats == "_"

        if fill_upos and has_xpos and (needs_upos or (fill_feats and needs_feats)):
            guess = self.guess_upos_feats(token.xpos, allow_partial=allow_partial)
            if guess:
                upos_guess, feats_guess = guess
                if needs_upos and upos_guess and upos_guess != "_":
                    token.upos = upos_guess
                    changed = True
                if fill_feats and needs_feats and feats_guess and feats_guess != "_":
                    token.feats = feats_guess
                    changed = True

        if fill_xpos and (not has_xpos):
            guess = self.guess_xpos(token.upos or "_", token.feats or "_", allow_partial=allow_partial)
            if guess:
                token.xpos = guess
                changed = True

        return changed

    def enrich_document(
        self,
        document: Document,
        *,
        fill_xpos: bool = True,
        fill_upos: bool = True,
        fill_feats: bool = True,
        allow_partial: bool = True,
    ) -> int:
        changes = 0
        for sentence in document.sentences:
            for token in sentence.tokens:
                if self.enrich_token(
                    token,
                    fill_xpos=fill_xpos,
                    fill_upos=fill_upos,
                    fill_feats=fill_feats,
                    allow_partial=allow_partial,
                ):
                    changes += 1
        return changes

    def _counter_for_xpos(self, xpos: str, *, allow_partial: bool) -> Optional[Counter]:
        if not xpos:
            return None
        if xpos in self.xpos_to_combo:
            return self.xpos_to_combo[xpos]

        lower = xpos.lower()
        if lower in self.xpos_lower_index:
            key = self.xpos_lower_index[lower]
            return self.xpos_to_combo.get(key)

        if not allow_partial:
            return None

        stripped = xpos.rstrip("0123456789.-_:")
        if stripped and stripped in self.xpos_to_combo:
            return self.xpos_to_combo[stripped]

        return None

    @staticmethod
    def _subset_of(target: Dict[str, str], candidate: Dict[str, str]) -> bool:
        for key, value in target.items():
            if key not in candidate or candidate[key] != value:
                return False
        return True


def build_tag_mapping_from_paths(paths: Iterable[Path]) -> TagMapping:
    mapping = TagMapping()
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        vocab = data.get("vocab", data)
        mapping._ingest_vocab(vocab)
    return mapping

