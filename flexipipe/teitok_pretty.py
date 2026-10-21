"""Pretty-printing TEITOK XML without changing its text (only when asked: --pretty-print).

In TEITOK the whitespace between tokens is part of the text, so indentation may only replace
whitespace whose meaning has been recorded first:

- ``join`` (default): a token not followed by whitespace before the next token (or text) gets
  ``join="right"``;
- ``c``: as ``join``, and every whitespace between elements becomes ``<c>…</c>`` holding the
  original characters (a newline stays a newline).

Then only elements with element-only content are indented — never the inside of a token, and never
an element that mixes text and elements (there whitespace stays as it is). teitok_text_check's
`reading` of the result equals that of the input.
"""

from __future__ import annotations

from typing import Optional

TOKEN_TAGS = {"tok", "dtok", "mtok"}
INDENT = "  "


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _ns_of(el) -> str:
    tag = el.tag if isinstance(el.tag, str) else ""
    return tag[: tag.index("}") + 1] if tag.startswith("{") else ""


def _text_root(root):
    for el in root.iter():
        if _local(el.tag) == "text":
            return el
    return root


def _mark_joins(text_root) -> None:
    """join="right" on every token that no whitespace follows before the next token or text."""
    last = None        # the last token
    space = False      # whitespace since it

    def outside(s: Optional[str]):
        nonlocal last, space
        if not s:
            return
        if last is not None and not space and not s[:1].isspace():
            last.set("join", "right")
        if s.strip():
            last = None
        space = s[-1:].isspace() or (space and s.strip() == "")

    def walk(el):
        nonlocal last, space
        tag = _local(el.tag)
        if tag == "tok":
            if last is not None and not space:
                last.set("join", "right")
            last, space = el, False
            return
        if tag == "c":
            space = True
            return
        if isinstance(el.tag, str):
            outside(el.text)
        for child in el:
            if isinstance(child.tag, str):
                walk(child)
            outside(child.tail)

    walk(text_root)


def _mixed(el) -> bool:
    """Text (not only whitespace) directly in `el`."""
    if el.text and el.text.strip():
        return True
    return any(child.tail and child.tail.strip() for child in el)


def _whitespace_to_c(el) -> None:
    """In element-only content: whitespace between elements as <c> elements."""
    from lxml import etree

    if _local(el.tag) in TOKEN_TAGS or _mixed(el):
        return
    children = [c for c in el if isinstance(c.tag, str)]
    if not children:
        return
    ns = _ns_of(el)
    if el.text:
        c = etree.Element(ns + "c")
        c.text = el.text
        el.text = None
        el.insert(0, c)
    for child in list(el):
        if child.tail and _local(child.tag) != "c":
            c = etree.Element(ns + "c")
            c.text = child.tail
            child.tail = None
            child.addnext(c)
        if isinstance(child.tag, str) and _local(child.tag) != "c":
            _whitespace_to_c(child)


def _indent(el, level: int) -> None:
    if _local(el.tag) in TOKEN_TAGS or _mixed(el):
        return
    children = list(el)
    if not children:
        return
    el.text = "\n" + INDENT * (level + 1)
    for child in children:
        if isinstance(child.tag, str):
            _indent(child, level + 1)
        child.tail = "\n" + INDENT * (level + 1)
    children[-1].tail = "\n" + INDENT * level


def pretty_print_tree(root, spacing: str = "join") -> None:
    """Indent the tree in place, the spacing between tokens recorded first (see the module doc)."""
    text_root = _text_root(root)
    for tok in text_root.iter():
        if _local(tok.tag) == "tok" and "SpaceAfter=No" in (tok.get("misc") or ""):
            tok.set("join", "right")
    _mark_joins(text_root)
    if spacing == "c":
        _whitespace_to_c(text_root)
    _indent(root, 0)


def pretty_print_xml_string(xml: str, spacing: str = "join") -> str:
    """`xml` pretty-printed (pretty_print_tree), with the XML declaration."""
    from lxml import etree

    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    root = etree.fromstring(xml.encode("utf-8"), parser)
    pretty_print_tree(root, spacing)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + etree.tostring(root, encoding="unicode") + "\n"
