"""Map TEITOK block elements to document sentences using global NLP offsets."""

from __future__ import annotations

import copy
import re
import sys
from typing import List, Optional, Sequence, Tuple

from .doc import Document, Sentence, Token

_WS_COLLAPSE = re.compile(r"\s+")


def sentence_nlp_surface(sentence: Sentence) -> str:
    if sentence.text and sentence.text.strip():
        return sentence.text
    parts: List[str] = []
    for tok in sentence.tokens:
        parts.append(tok.form)
        if tok.space_after is not False:
            parts.append(" ")
    text = "".join(parts)
    if parts and parts[-1] == " ":
        text = text[:-1]
    return text


def _compact(s: str) -> str:
    return _WS_COLLAPSE.sub("", s)


def _sentence_tokens(sentence: Sentence) -> List[Token]:
    """Parent tokens only (skip MWT subtokens after the first)."""
    subtoken_ids = set()
    for token in sentence.tokens:
        if token.is_mwt and token.subtokens:
            for st in token.subtokens[1:]:
                if st.id:
                    subtoken_ids.add(st.id)
    return [t for t in sentence.tokens if not (t.id and t.id in subtoken_ids)]


def find_tokens_at(
    nlp_text: str, tokens: Sequence[Token], start: int = 0
) -> int:
    """Find where token forms appear consecutively in nlp_text (whitespace-flexible)."""
    if not tokens:
        return start
    first = tokens[0].form
    if not first:
        return -1
    pos = nlp_text.find(first, start)
    while 0 <= pos < len(nlp_text):
        idx = pos + len(first)
        ok = True
        for tok in tokens[1:]:
            while idx < len(nlp_text) and nlp_text[idx].isspace():
                idx += 1
            if idx >= len(nlp_text) or not nlp_text.startswith(tok.form, idx):
                ok = False
                break
            idx += len(tok.form)
        if ok:
            return pos
        pos = nlp_text.find(first, pos + 1)
    return -1


def _token_span_end(nlp_text: str, pos: int, tokens: Sequence[Token]) -> int:
    end = pos
    for tok in tokens:
        while end < len(nlp_text) and nlp_text[end].isspace():
            end += 1
        end += len(tok.form)
    return end


def _map_compact_to_raw(text: str, base: int, compact_start: int, compact_len: int) -> int:
    """Map position in compact(text[base:]) to index in text."""
    c = 0
    i = base
    while i < len(text) and c < compact_start:
        if not text[i].isspace():
            c += 1
        i += 1
    return i


def build_sentence_nlp_offsets(
    document: Document,
    nlp_text: str,
    *,
    verbose: bool = False,
) -> List[Tuple[int, int, int]]:
    """Return (sent_idx, nlp_start, nlp_end) by walking tokens left-to-right in nlp_text."""
    spans: List[Tuple[int, int, int]] = []
    cursor = 0
    failed = 0

    for idx, sent in enumerate(document.sentences):
        tokens = _sentence_tokens(sent)
        if tokens:
            pos = find_tokens_at(nlp_text, tokens, cursor)
            if pos < 0:
                surface = sentence_nlp_surface(sent).strip()
                if surface:
                    pos = nlp_text.find(surface, cursor)
                if pos < 0 and surface:
                    compact_h = _compact(nlp_text[cursor:])
                    compact_n = _compact(surface)
                    rel = compact_h.find(compact_n) if compact_n else -1
                    if rel >= 0:
                        pos = _map_compact_to_raw(nlp_text, cursor, rel, len(compact_n))
                if pos < 0:
                    failed += 1
                    if verbose:
                        first = tokens[0].form if tokens else "?"
                        print(
                            f"[flexipipe] sentence {idx}: could not anchor at cursor {cursor}, first token {first!r}",
                            file=sys.stderr,
                        )
                    continue
                end = pos + len(surface) if surface else _token_span_end(nlp_text, pos, tokens)
            else:
                end = _token_span_end(nlp_text, pos, tokens)
        else:
            surface = sentence_nlp_surface(sent).strip()
            if not surface:
                continue
            pos = nlp_text.find(surface, cursor)
            if pos < 0:
                compact_h = _compact(nlp_text[cursor:])
                compact_n = _compact(surface)
                rel = compact_h.find(compact_n) if compact_n else -1
                if rel < 0:
                    failed += 1
                    if verbose:
                        print(
                            f"[flexipipe] sentence {idx}: could not anchor empty-token sentence",
                            file=sys.stderr,
                        )
                    continue
                pos = _map_compact_to_raw(nlp_text, cursor, rel, len(compact_n))
            end = pos + len(surface)

        spans.append((idx, pos, end))
        cursor = end
        while cursor < len(nlp_text) and nlp_text[cursor].isspace():
            cursor += 1

    if verbose and failed:
        print(
            f"[flexipipe] Anchored {len(spans)}/{len(document.sentences)} sentences in NLP text ({failed} failed)",
            file=sys.stderr,
        )
    return spans


def assign_sentence_spans(
    document: Document, nlp_text: str, *, verbose: bool = False
) -> List[Tuple[int, int, int]]:
    """Return (sent_idx, nlp_start, nlp_end) for each sentence in order."""
    return build_sentence_nlp_offsets(document, nlp_text, verbose=verbose)


def build_block_sentence_map(
    document: Document,
    nlp_text: str,
    block_nlp_ranges: Sequence[Tuple[int, int]],
    *,
    verbose: bool = False,
) -> List[List[int]]:
    """Map each block to sentence indices whose NLP span overlaps the block range."""
    cached = document.meta.get("_teitok_sentence_nlp_spans")
    if cached:
        sent_spans = cached
    else:
        sent_spans = build_sentence_nlp_offsets(document, nlp_text, verbose=verbose)
        document.meta["_teitok_sentence_nlp_spans"] = sent_spans

    block_map: List[List[int]] = [[] for _ in block_nlp_ranges]

    for idx, s, e in sent_spans:
        for bi, (bstart, bend) in enumerate(block_nlp_ranges):
            if bend <= bstart:
                continue
            if s < bend and e > bstart:
                block_map[bi].append(idx)

    return block_map


def clip_sentences_to_nlp_range(
    document: Document,
    sentence_indices: Sequence[int],
    nlp_text: str,
    block_start: int,
    block_end: int,
) -> Tuple[List[Sentence], List[int], List[int]]:
    """
    Restrict sentences to tokens that lie in [block_start, block_end).

    Returns (clipped_sentences, source_indices, fully_inside_indices).
    fully_inside_indices are sentences wholly inside the block (safe to mark used).
    """
    sent_spans_list = document.meta.get("_teitok_sentence_nlp_spans")
    if not sent_spans_list:
        sent_spans_list = build_sentence_nlp_offsets(document, nlp_text)
    span_by_idx = {i: (s, e) for i, s, e in sent_spans_list}

    clipped: List[Sentence] = []
    sources: List[int] = []
    fully_inside: List[int] = []

    for idx in sentence_indices:
        sent = document.sentences[idx]
        span = span_by_idx.get(idx)
        if span is None:
            clipped.append(sent)
            sources.append(idx)
            fully_inside.append(idx)
            continue
        s_start, s_end = span
        if s_start >= block_start and s_end <= block_end:
            clipped.append(sent)
            sources.append(idx)
            fully_inside.append(idx)
            continue

        tokens = _sentence_tokens(sent)
        if not tokens:
            if s_start < block_end and s_end > block_start:
                clipped.append(copy.copy(sent))
                sources.append(idx)
            continue

        anchor = max(0, min(s_start, block_start) - 20)
        pos = find_tokens_at(nlp_text, tokens, anchor)
        if pos < 0:
            pos = find_tokens_at(nlp_text, tokens, s_start)
        if pos < 0:
            continue

        kept_parents: List[Token] = []
        cursor = pos
        for tok in tokens:
            while cursor < len(nlp_text) and nlp_text[cursor].isspace():
                cursor += 1
            tok_start = cursor
            tok_end = cursor + len(tok.form)
            if tok_end <= block_start:
                cursor = tok_end
                continue
            if tok_start >= block_end:
                break
            if tok_start < block_end and tok_end > block_start:
                kept_parents.append(tok)
            cursor = tok_end

        if not kept_parents:
            continue

        kept_ids = {id(t) for t in kept_parents}
        subtoken_ids: set = set()
        for t in kept_parents:
            if t.is_mwt and t.subtokens:
                for st in t.subtokens[1:]:
                    if st.id:
                        subtoken_ids.add(st.id)

        new_tokens: List[Token] = []
        for t in sent.tokens:
            if id(t) in kept_ids:
                new_tokens.append(t)
            elif t.id and t.id in subtoken_ids:
                new_tokens.append(t)

        new_sent = copy.copy(sent)
        new_sent.tokens = new_tokens
        clipped.append(new_sent)
        sources.append(idx)

    return clipped, sources, fully_inside


def find_sentences_for_block_nlp_slice(
    document: Document,
    nlp_text: str,
    block_start: int,
    block_end: int,
    *,
    exclude: Optional[set] = None,
) -> List[int]:
    """Fallback: locate sentences by first-token search inside a block NLP slice."""
    exclude = exclude or set()
    block_slice = nlp_text[block_start:block_end]
    if not block_slice.strip():
        return []

    sent_spans = document.meta.get("_teitok_sentence_nlp_spans") or build_sentence_nlp_offsets(
        document, nlp_text
    )
    indices: List[int] = []
    for idx, s, e in sent_spans:
        if idx in exclude:
            continue
        if s >= block_end or e <= block_start:
            continue
        if block_start <= s < block_end or block_start < e <= block_end:
            indices.append(idx)
    if indices:
        return indices

    for idx, sent in enumerate(document.sentences):
        if idx in exclude:
            continue
        tokens = _sentence_tokens(sent)
        if not tokens:
            continue
        first = tokens[0].form
        if first and block_slice.find(first) >= 0:
            indices.append(idx)
    return indices
