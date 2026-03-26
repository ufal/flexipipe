"""
Minidom-based rebuild of TEITOK blocks with <s> and <tok> inserted.

In DOM (minidom), all text is in explicit Text nodes; there is no .tail.
So we get a linear list of (start, end, text_node, path) in document order
and can insert <s>/<tok> by splitting text nodes and wrapping ranges,
without the tail-handling complexity of lxml/ElementTree.
"""

from __future__ import annotations

import xml.dom.minidom as minidom
from xml.dom import Node
from typing import Any, Dict, List, Optional, Tuple

# Document/Sentence/Token from doc
try:
    from .doc import Document, Sentence, Token
except ImportError:
    Document = None  # type: ignore
    Sentence = None
    Token = None  # type: ignore


def _text_content(node: Any) -> str:
    """Recursive text content of a DOM node (no tail; children only)."""
    if node.nodeType == Node.TEXT_NODE:
        return node.data or ""
    if node.nodeType == Node.CDATA_SECTION_NODE:
        return node.data or ""
    out = []
    for child in node.childNodes:
        out.append(_text_content(child))
    return "".join(out)


def _build_segments(
    block_node: Any,
    plaintext: str,
) -> List[Tuple[int, int, Any, List[Tuple[str, Dict[str, str]]]]]:
    """
    Walk block in document order; for each Text node record (start, end, node, path).
    path = list of (tag_name, attrib_dict) from block down to parent of text node.
    """
    segments: List[Tuple[int, int, Any, List[Tuple[str, Dict[str, str]]]]] = []
    pos = 0

    def walk(node: Any, path: List[Tuple[str, Dict[str, str]]]) -> None:
        nonlocal pos
        if node.nodeType == Node.TEXT_NODE or node.nodeType == Node.CDATA_SECTION_NODE:
            data = node.data or ""
            n = len(data)
            if n > 0:
                segments.append((pos, pos + n, node, path))
            pos += n
            return
        if node.nodeType != Node.ELEMENT_NODE:
            return
        tag = node.tagName if hasattr(node, "tagName") else node.nodeName
        attrib: Dict[str, str] = {}
        if hasattr(node, "attributes") and node.attributes:
            for i in range(node.attributes.length):
                a = node.attributes.item(i)
                if a and a.name and a.value is not None:
                    attrib[a.name] = a.value
        path_here = path + [(tag, attrib)]
        for child in node.childNodes:
            walk(child, path_here)

    walk(block_node, [])
    return segments


def _segment_tag(tag: str, attrib: Dict[str, str], doc: Any) -> Any:
    """Create an element with optional namespace (minidom)."""
    if "}" in tag:
        # lxml-style {uri}localname
        uri, _, local = tag.partition("}")
        return doc.createElementNS(uri or None, local)
    elem = doc.createElement(tag)
    for k, v in attrib.items():
        if k == "xmlns" or k.startswith("xmlns:"):
            continue
        elem.setAttribute(k, v)
    return elem


def rebuild_block_xml_minidom(
    block_xml_string: str,
    plaintext: str,
    token_positions: List[Tuple[int, int, Any]],
    sentence_positions: List[Tuple[int, int, Any]],
    start_tok_id: int = 1,
    start_sent_idx: int = 0,
    ord_to_tokid: Optional[Dict[int, str]] = None,
    resolve_attr: Optional[Any] = None,
    set_attr: Optional[Any] = None,
) -> str:
    """
    Rebuild a single block's XML string by inserting <s> and <tok> using minidom.
    All text is in #text nodes; no .tail, so insertion is straightforward.

    Args:
        block_xml_string: Serialized block element (e.g. <p>...</p>).
        plaintext: Extracted plain text for the block.
        token_positions: List of (start, end, Token).
        sentence_positions: List of (start, end, Sentence).
        start_tok_id, start_sent_idx: Counters for ids.
        ord_to_tokid: Optional map ord -> tokid for head conversion.
        resolve_attr, set_attr: Optional (node, attr, value) for TEITOK attr mapping.

    Returns:
        Tuple of (new_xml_string, tokens_used, sentences_used).
    """
    doc = minidom.parseString(block_xml_string)
    block = doc.documentElement
    segments = _build_segments(block, plaintext)
    if not segments and not token_positions:
        return block_xml_string, 0, 0

    # Build sorted boundary events: (pos, 'open_s'|'close_s'|'open_t'|'close_t', payload)
    events: List[Tuple[int, str, Any]] = []
    for s_start, s_end, sent in sentence_positions:
        events.append((s_start, "open_s", sent))
        events.append((s_end, "close_s", sent))
    for t_start, t_end, tok in token_positions:
        events.append((t_start, "open_t", tok))
        events.append((t_end, "close_t", tok))
    events.sort(key=lambda e: (e[0], 0 if e[1].startswith("open") else 1))

    # New document we'll build
    new_doc = minidom.getDOMImplementation().createDocument(None, None, None)
    block_tag = block.tagName if hasattr(block, "tagName") else block.nodeName
    block_attrib: Dict[str, str] = {}
    if hasattr(block, "attributes") and block.attributes:
        for i in range(block.attributes.length):
            a = block.attributes.item(i)
            if a and a.name and a.value is not None:
                block_attrib[a.name] = a.value
    new_block = _segment_tag(block_tag, block_attrib, new_doc)
    new_doc.appendChild(new_block)

    stack: List[Any] = [new_block]
    path_stack: List[List[Tuple[str, Dict[str, str]]]] = [[]]
    seg_idx = 0
    ev_idx = 0
    global_tok_id = start_tok_id
    sent_idx = start_sent_idx
    current_sent_obj: Optional[Any] = None
    sent_token_ids: List[str] = []
    tok_id_map: Dict[int, str] = dict(ord_to_tokid or {})

    def ensure_path(path: List[Tuple[str, Dict[str, str]]]) -> None:
        """Open elements so stack top has path; close then open as needed."""
        nonlocal stack, path_stack
        # Close until we match a prefix of path
        while len(path_stack) > 1 and (len(path) < len(path_stack) or path[: len(path_stack)] != path_stack):
            stack.pop()
            path_stack.pop()
        # Open new path elements
        for i in range(len(path_stack), len(path)):
            tag, attrib = path[i]
            elem = _segment_tag(tag, attrib, new_doc)
            stack[-1].appendChild(elem)
            stack.append(elem)
            path_stack.append(path[: i + 1])

    def flush_events_until(pos: int) -> None:
        nonlocal ev_idx, global_tok_id, sent_idx, current_sent_obj, sent_token_ids
        while ev_idx < len(events) and events[ev_idx][0] <= pos:
            p, kind, payload = events[ev_idx]
            ev_idx += 1
            if kind == "open_s":
                current_sent_obj = payload
                s_elem = new_doc.createElement("s")
                s_elem.setAttribute("id", f"s-{sent_idx + 1}")
                if getattr(payload, "text", None):
                    s_elem.setAttribute("text", payload.text)
                stack[-1].appendChild(s_elem)
                stack.append(s_elem)
                sent_token_ids = []
            elif kind == "close_s":
                if stack and stack[-1].tagName == "s":
                    if sent_token_ids:
                        stack[-1].setAttribute("corresp", " ".join(f"#{t}" for t in sent_token_ids))
                    stack.pop()
                sent_idx += 1
                current_sent_obj = None
            elif kind == "open_t":
                tok = payload
                tok_elem = new_doc.createElement("tok")
                tok_id = f"w-{global_tok_id}"
                tok_id_map[getattr(tok, "id", None) or global_tok_id] = tok_id
                tok_elem.setAttribute("id", tok_id)
                sent_token_ids.append(tok_id)
                # Text is added by segment iteration (append to stack top)
                stack[-1].appendChild(tok_elem)
                stack.append(tok_elem)
                global_tok_id += 1
            elif kind == "close_t":
                if stack and (stack[-1].tagName == "tok" or getattr(stack[-1], "tagName", None) == "tok"):
                    stack.pop()

    # Iterate segments and emit text (splitting at token/sentence boundaries)
    for seg_start, seg_end, _text_node, path in segments:
        flush_events_until(seg_start)
        ensure_path(path)
        # Boundaries inside this segment (positions where we open/close)
        boundaries = [seg_start]
        j = ev_idx
        while j < len(events):
            p, _, _ = events[j]
            if p <= seg_start:
                j += 1
                continue
            if p >= seg_end:
                break
            if p != boundaries[-1]:
                boundaries.append(p)
            j += 1
        boundaries.append(seg_end)
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            if a >= b:
                continue
            flush_events_until(a)
            text = plaintext[a:b]
            if text:
                stack[-1].appendChild(new_doc.createTextNode(text))
            flush_events_until(b)
    flush_events_until(len(plaintext) + 1)

    # Pop any remaining path (should be back to new_block)
    while len(stack) > 1:
        stack.pop()

    # Head conversion: ord -> tokid on <tok> elements
    def walk_set_heads(node: Any) -> None:
        if node.nodeType != Node.ELEMENT_NODE:
            return
        tag = getattr(node, "tagName", None) or node.nodeName
        if tag == "tok" and node.hasAttribute("head"):
            head = node.getAttribute("head")
            if head and head.isdigit():
                ord_val = int(head)
                tid = tok_id_map.get(ord_val)
                if tid:
                    node.setAttribute("head", tid)
                elif ord_val == 0:
                    node.removeAttribute("head")
        for c in node.childNodes:
            walk_set_heads(c)

    walk_set_heads(new_block)
    tokens_used = global_tok_id - start_tok_id
    sentences_used = sent_idx - start_sent_idx
    return new_doc.toxml(), tokens_used, sentences_used
