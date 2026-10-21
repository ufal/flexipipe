"""Has a TEITOK file's text changed? Used before every writeback (xml_safe_write).

In TEITOK the text is the XML's text content: annotating adds markup (<s>, <tok>, attributes) but
must leave the characters — the spacing between tokens above all — as they were.

- :func:`text_change` (no pretty-printing): the text content of <text> must be identical,
  character for character, whitespace included.
- :func:`spacing_change` (pretty-printed output): whitespace may differ, but the text as TEITOK
  reads it (:func:`reading`: whitespace runs as one space, none after ``join="right"``, ``<c>`` as
  its content) must stay the same — every token's form and whether a space follows it.

Both return None when nothing changed, else a short description with the place.
"""

from __future__ import annotations

import unicodedata
from typing import List, Optional

from lxml import etree

TOKEN_TAGS = {"tok"}
# inside a token everything is the token's text (dtok, ex, …)


def _local(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _parse(data: bytes):
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True, remove_blank_text=False)
    return etree.fromstring(data, parser)


def _text_root(root):
    """The <text> element (TEI), else the whole document."""
    if _local(root.tag) == "text":
        return root
    for el in root.iter():
        if _local(el.tag) == "text":
            return el
    return root


def _norm(s: str, normalization: Optional[str]) -> str:
    return unicodedata.normalize(normalization.upper(), s) if normalization else s


def plain_text(data: bytes) -> str:
    """The text content of <text>: every text node in document order (comments, PIs left out)."""
    parts: List[str] = []

    def walk(el):
        if isinstance(el.tag, str) and el.text:
            parts.append(el.text)
        for child in el:
            if isinstance(child.tag, str):
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(_text_root(_parse(data)))
    return "".join(parts)


def _where(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    show = lambda s: repr(s[max(0, i - 25): i + 25])
    return f"at character {i}: before {show(a)}, after {show(b)}"


def text_change(original: bytes, new: bytes, normalization: Optional[str] = None) -> Optional[str]:
    """Why the text content of <text> differs (no pretty-printing), or None."""
    a = _norm(plain_text(original), normalization)
    b = _norm(plain_text(new), normalization)
    if a == b:
        return None
    return f"the text changed {_where(a, b)}"


def reading(data: bytes) -> str:
    """The text as TEITOK reads it: token forms and the text outside tokens, every whitespace run
    outside tokens as one space, none after a token with join="right"; <c> counts as its content."""
    out: List[str] = []
    pending = False   # whitespace seen since the last text
    suppress = False  # the last token has join="right"

    def emit(text: str):
        nonlocal pending, suppress
        if out and pending and not suppress:
            out.append(" ")
        out.append(text)
        pending = False
        suppress = False

    def outside(s: Optional[str]):
        nonlocal pending
        if not s:
            return
        for piece in _split_ws(s):
            if piece.isspace():
                pending = True
            else:
                emit(piece)

    def walk(el):
        nonlocal suppress
        tag = _local(el.tag)
        if tag in TOKEN_TAGS:
            emit("".join(el.itertext()))
            suppress = (el.get("join") or "").lower() == "right"
            return
        if tag == "c":
            outside("".join(el.itertext()) or " ")
            return
        outside(el.text)
        for child in el:
            if isinstance(child.tag, str):
                walk(child)
            outside(child.tail)

    walk(_text_root(_parse(data)))
    return "".join(out)


def _split_ws(s: str) -> List[str]:
    import re

    return [p for p in re.split(r"(\s+)", s) if p]


def spacing_change(original: bytes, new: bytes, normalization: Optional[str] = None) -> Optional[str]:
    """Why the text as TEITOK reads it differs (pretty-printed output), or None."""
    a = _norm(reading(original), normalization)
    b = _norm(reading(new), normalization)
    if a == b:
        return None
    return f"the text or the spacing between tokens changed {_where(a, b)}"
