"""Optional TEITOK XML layout normalization before xmltokenizer extract.

Inserts ``tail`` text on block elements only when they have no tail yet and
the next sibling is another block element — e.g. ``</p><p>`` becomes
``</p>\\n<p>`` on serialize without regex and without changing element
``.text`` (tokenization plaintext is unchanged until xmltokenizer handles
paragraph breaks in its own layer).
"""

from __future__ import annotations

from typing import AbstractSet, Optional

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# Match common TEITOK block tags (same defaults as insert_tokens).
DEFAULT_BLOCK_TAGS: frozenset[str] = frozenset(
    {"p", "div", "head", "lg", "ab", "u", "speaker", "cell", "row"}
)


def _local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag if isinstance(tag, str) else ""


def _is_block(elem: ET.Element, block_tags: AbstractSet[str]) -> bool:
    return _local_tag(elem).lower() in block_tags


def ensure_block_element_tails(
    root: ET.Element,
    block_tags: Optional[AbstractSet[str]] = None,
    *,
    separator: str = "\n",
) -> int:
    """Set ``tail`` on block elements that have none before a block sibling.

    Returns the number of elements updated.
    """
    if not separator:
        raise ValueError("separator must be non-empty")
    tags = block_tags if block_tags is not None else DEFAULT_BLOCK_TAGS
    tags_lower = {t.lower() for t in tags}
    updated = 0

    for parent in root.iter():
        children = list(parent)
        for i, child in enumerate(children):
            if not _is_block(child, tags_lower):
                continue
            if i + 1 >= len(children):
                continue
            if not _is_block(children[i + 1], tags_lower):
                continue
            if child.tail is not None and child.tail != "":
                continue
            child.tail = separator
            updated += 1

    return updated


def layout_normalize_deactivated_bytes(
    raw_de: bytes,
    block_tags: Optional[AbstractSet[str]] = None,
    *,
    separator: str = "\n",
) -> bytes:
    """Parse deactivated XML, add block tails where missing, serialize to bytes."""
    try:
        parser = ET.XMLParser(strip_cdata=False, remove_blank_text=False)
        root = ET.fromstring(raw_de, parser)
    except TypeError:
        root = ET.fromstring(raw_de)

    ensure_block_element_tails(root, block_tags, separator=separator)

    if hasattr(ET, "tostring"):
        out = ET.tostring(
            root,
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=False,
        )
    else:
        out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return bytes(out)


def layout_normalize_teitok_file_bytes(
    raw: bytes,
    *,
    deactivate,
    block_tags: Optional[AbstractSet[str]] = None,
    separator: str = "\n",
) -> tuple[bytes, object]:
    """Deactivate, layout-normalize, return (canonical_deactivated_bytes, ns_transform)."""
    raw_de, ns_transform = deactivate(raw)
    canonical = layout_normalize_deactivated_bytes(
        raw_de, block_tags=block_tags, separator=separator
    )
    return canonical, ns_transform
