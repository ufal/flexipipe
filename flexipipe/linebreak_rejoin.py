"""Detect and apply line-break / hyphenation rejoin for historic TEI tokenization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_LETTER_RUN = re.compile(r"[\w\u00C0-\u024F]+", re.UNICODE)
_WHITESPACE_ONLY = re.compile(r"^\s*$")


@dataclass
class RejoinSpan:
    """A display-plaintext span merged for NLP, with inner XML preserved on insert."""

    display_start: int
    display_end: int  # exclusive
    merged_form: str
    nlp_start: int = 0
    nlp_end: int = 0  # exclusive
    # Display interval omitted from NLP (hyphen, whitespace, markup gaps between parts)
    removed_display_start: int = 0
    removed_display_end: int = 0  # exclusive; equals display_end when whole tail is removed

    @property
    def display_shrink(self) -> int:
        return (self.display_end - self.display_start) - (self.nlp_end - self.nlp_start)


def _local_tag(elem: Any) -> str:
    tag = getattr(elem, "tag", "") or ""
    if not isinstance(tag, str):
        return ""
    return tag.split("}")[-1].lower()


def _lb_break_no(elem: Any) -> bool:
    if elem is None:
        return False
    break_val = elem.get("break")
    if break_val is None:
        for key, val in elem.attrib.items():
            if key.endswith("}break") or key == "break":
                break_val = val
                break
    if break_val is None:
        return False
    return str(break_val).strip().lower() == "no"


def _letter_run_before(plaintext: str, pos: int) -> Optional[Tuple[int, int, str]]:
    if pos <= 0:
        return None
    matches = list(_LETTER_RUN.finditer(plaintext[:pos]))
    if not matches:
        return None
    m = matches[-1]
    return m.start(), m.end(), m.group(0)


def _letter_run_after(plaintext: str, pos: int) -> Optional[Tuple[int, int, str]]:
    if pos >= len(plaintext):
        return None
    m = _LETTER_RUN.search(plaintext[pos:])
    if not m:
        return None
    return pos + m.start(), pos + m.end(), m.group(0)


def _has_hyphen_bridge(plaintext: str, b_end: int, a_start: int) -> bool:
    mid = plaintext[b_end:a_start]
    if _WHITESPACE_ONLY.match(mid):
        return False
    compact = re.sub(r"\s+", "", mid)
    return compact in ("-", "–", "—") or "-" in compact or "–" in compact


def _markup_blocks_rejoin(
    markup: Sequence[Dict[str, Any]], gap_start: int, gap_end: int
) -> bool:
    """True if a column/page break element sits between two letter runs."""
    for m in markup:
        pos = m.get("start")
        if pos is None:
            continue
        if (m.get("name") or "").lower() not in ("cb", "pb"):
            continue
        if gap_start <= pos <= gap_end:
            return True
    return False


def _should_rejoin(
    before: str,
    after: str,
    *,
    explicit_break_no: bool,
    has_hyphen_bridge: bool,
) -> bool:
    if not before or not after:
        return False
    if explicit_break_no:
        return True
    if not has_hyphen_bridge and not before.rstrip().endswith(("-", "–")):
        return False
    if after[0].islower():
        return True
    if len(after) <= 12:
        return True
    return False


def _merge_forms(before: str, after: str) -> str:
    left = before.rstrip("-–— \t")
    return left + after


def detect_rejoin_spans(
    plaintext: str,
    markup: Sequence[Dict[str, Any]],
) -> List[RejoinSpan]:
    """Find word splits across <lb/> that should become one NLP token."""
    if not plaintext:
        return []

    spans: List[RejoinSpan] = []
    used: List[Tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= us or s >= ue) for us, ue in used)

    lb_entries = [
        m
        for m in markup
        if (m.get("name") or "").lower() in ("lb", "pb")
        and isinstance(m.get("start"), int)
        and m.get("start") == m.get("end")
    ]
    lb_entries.sort(key=lambda m: m["start"])

    for entry in lb_entries:
        pos = entry["start"]
        elem = entry.get("element")
        explicit = _lb_break_no(elem) if _local_tag(elem) == "lb" else False

        before_run = _letter_run_before(plaintext, pos)
        after_run = _letter_run_after(plaintext, pos)
        if not before_run or not after_run:
            continue

        b_start, b_end, b_text = before_run
        a_start, a_end, a_text = after_run
        hyphen = _has_hyphen_bridge(plaintext, b_end, a_start) or b_text.endswith(("-", "–"))

        if _markup_blocks_rejoin(markup, b_end, a_start):
            continue

        if not _should_rejoin(
            b_text, a_text, explicit_break_no=explicit, has_hyphen_bridge=hyphen
        ):
            continue

        display_start, display_end = b_start, a_end
        if overlaps(display_start, display_end):
            continue

        merged = _merge_forms(b_text, a_text)
        removed_start = display_start + len(merged)
        spans.append(
            RejoinSpan(
                display_start=display_start,
                display_end=display_end,
                merged_form=merged,
                removed_display_start=removed_start,
                removed_display_end=display_end,
            )
        )
        used.append((display_start, display_end))

    spans.sort(key=lambda s: s.display_start)
    return spans


def build_nlp_plaintext(
    display_plaintext: str,
    rejoin_spans: Sequence[RejoinSpan],
) -> Tuple[str, List[RejoinSpan]]:
    """Build NLP plaintext and assign nlp_start/nlp_end on spans."""
    if not rejoin_spans:
        return display_plaintext, []

    spans = sorted(rejoin_spans, key=lambda s: s.display_start)
    parts: List[str] = []
    nlp_pos = 0
    cursor = 0
    out: List[RejoinSpan] = []

    for span in spans:
        if span.display_start < cursor:
            continue
        parts.append(display_plaintext[cursor : span.display_start])
        nlp_pos += span.display_start - cursor
        span.nlp_start = nlp_pos
        parts.append(span.merged_form)
        nlp_pos += len(span.merged_form)
        span.nlp_end = nlp_pos
        if span.removed_display_end <= span.removed_display_start:
            span.removed_display_start = span.display_start + len(span.merged_form)
            span.removed_display_end = span.display_end
        cursor = span.display_end
        out.append(span)

    parts.append(display_plaintext[cursor:])
    return "".join(parts), out


def nlp_pos_to_display(nlp_pos: int, rejoin_spans: Sequence[RejoinSpan]) -> int:
    """Map a single NLP offset to display plaintext (inverse of build_nlp_plaintext)."""
    if not rejoin_spans:
        return nlp_pos

    offset = 0
    for span in sorted(rejoin_spans, key=lambda s: s.nlp_start):
        if nlp_pos < span.nlp_start:
            return nlp_pos + offset
        if nlp_pos < span.nlp_end:
            return span.display_start + (nlp_pos - span.nlp_start)
        if nlp_pos == span.nlp_end:
            return span.display_end
        offset = span.display_end - span.nlp_end

    return nlp_pos + offset


def nlp_span_to_display(
    nlp_start: int,
    nlp_end: int,
    rejoin_spans: Sequence[RejoinSpan],
) -> Tuple[int, int]:
    """Map NLP [start,end) to display plaintext coordinates."""
    if not rejoin_spans:
        return nlp_start, nlp_end

    ds = nlp_pos_to_display(nlp_start, rejoin_spans)
    de = nlp_pos_to_display(nlp_end, rejoin_spans)
    if de < ds:
        de = ds
    return ds, de


def slice_rejoin_spans_for_block(
    block_display_range: Tuple[int, int],
    block_nlp_range: Tuple[int, int],
    global_spans: Sequence[RejoinSpan],
) -> List[RejoinSpan]:
    """Return block-local rejoin spans from extract-time global spans."""
    bds, bde = block_display_range
    bns, bne = block_nlp_range
    out: List[RejoinSpan] = []
    for span in global_spans:
        if span.display_end <= bds or span.display_start >= bde:
            continue
        if span.nlp_end <= bns or span.nlp_start >= bne:
            continue
        out.append(
            RejoinSpan(
                display_start=max(span.display_start, bds) - bds,
                display_end=min(span.display_end, bde) - bds,
                merged_form=span.merged_form,
                nlp_start=max(span.nlp_start, bns) - bns,
                nlp_end=min(span.nlp_end, bne) - bns,
                removed_display_start=max(span.removed_display_start, bds) - bds,
                removed_display_end=min(span.removed_display_end, bde) - bds,
            )
        )
    return out


def apply_rejoin_to_token_positions(
    token_positions: List[Tuple[int, int, Any]],
    rejoin_spans: Sequence[RejoinSpan],
) -> List[Tuple[int, int, Any]]:
    if not rejoin_spans:
        return token_positions
    out: List[Tuple[int, int, Any]] = []
    for s, e, t in token_positions:
        ds, de = nlp_span_to_display(s, e, rejoin_spans)
        out.append((ds, de, t))
    return out


def merged_form_for_display_span(
    display_start: int,
    display_end: int,
    rejoin_spans: Sequence[RejoinSpan],
) -> Optional[str]:
    for span in rejoin_spans:
        if span.display_start == display_start and span.display_end == display_end:
            return span.merged_form
    return None
