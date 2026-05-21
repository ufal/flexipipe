"""Insert TEITOK <name> wrappers around entity token spans."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from .doc import Document, Entity, Sentence


def _local_tag(elem: Any) -> str:
    tag = getattr(elem, "tag", "") or ""
    if not isinstance(tag, str):
        return ""
    return tag.split("}")[-1].lower()


def _tok_ord(elem: ET.Element) -> Optional[int]:
    for key in ("ord", "id"):
        val = elem.get(key)
        if val is None:
            continue
        if key == "ord":
            try:
                return int(val)
            except ValueError:
                pass
        if key == "id" and val.startswith("w-"):
            try:
                return int(val[2:].split(".")[0])
            except ValueError:
                pass
    return None


def _find_sentence_elements(root: ET.Element) -> List[ET.Element]:
    return [n for n in root.iter() if _local_tag(n) == "s"]


def wrap_names_in_sentence(s_elem: ET.Element, sentence: Sentence) -> None:
    entities = list(sentence.entities or [])
    if not entities:
        return

    tok_elems: List[ET.Element] = [c for c in list(s_elem) if _local_tag(c) == "tok"]
    if not tok_elems:
        return

    ord_to_elem: Dict[int, ET.Element] = {}
    for elem in tok_elems:
        o = _tok_ord(elem)
        if o is not None:
            ord_to_elem[o] = elem

    parent_map: Dict[ET.Element, ET.Element] = {c: s_elem for c in tok_elems}

    for ent in sorted(entities, key=lambda e: (e.start, e.end)):
        group = [ord_to_elem[o] for o in range(ent.start, ent.end + 1) if o in ord_to_elem]
        if not group:
            continue
        if all(_local_tag(parent_map.get(t, s_elem)) == "name" for t in group):
            continue

        name_elem = ET.Element("name")
        if ent.label:
            name_elem.set("type", ent.label)
        if ent.text:
            name_elem.set("text", ent.text)
        for key, value in (ent.attrs or {}).items():
            if key and value is not None:
                name_elem.set(str(key), str(value))

        first = group[0]
        parent = parent_map.get(first, s_elem)
        children = list(parent)
        insert_at = children.index(first)
        parent.insert(insert_at, name_elem)
        for tok in group:
            if tok in children:
                parent.remove(tok)
            name_elem.append(tok)
            parent_map[tok] = name_elem


def apply_name_wrappers_to_tree(root: ET.Element, document: Document) -> None:
    s_elems = _find_sentence_elements(root)
    for idx, sent in enumerate(document.sentences):
        if idx >= len(s_elems):
            break
        wrap_names_in_sentence(s_elems[idx], sent)
