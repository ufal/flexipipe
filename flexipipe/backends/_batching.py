"""Splitting a document for REST backends that refuse large payloads (UDPipe, UDMorph, NameTag)."""

from __future__ import annotations

from typing import List

from ..doc import Document


def split_document(document: Document, batch_size: int) -> List[Document]:
    """The document's sentences in batches of `batch_size`, each a document with the same meta."""
    if not document.sentences or batch_size <= 0 or len(document.sentences) <= batch_size:
        return [document]
    batches: List[Document] = []
    for start in range(0, len(document.sentences), batch_size):
        batches.append(Document(
            id=f"{document.id or 'doc'}-batch-{len(batches) + 1}",
            meta=dict(document.meta),
            sentences=[*document.sentences[start:start + batch_size]],
        ))
    return batches


def split_text(text: str, max_chars: int) -> List[str]:
    """Raw text in parts of at most `max_chars`, split at paragraph barriers (blank lines), which a segmenter
    does not cross anyway; a single paragraph larger than that at line breaks."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    current: List[str] = []
    size = 0
    for block in (b for para in text.split("\n\n") for b in _lines_if_large(para, max_chars)):
        if current and size + len(block) + 2 > max_chars:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(block)
        size += len(block) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts


def _lines_if_large(paragraph: str, max_chars: int) -> List[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out: List[str] = []
    current = ""
    for line in paragraph.split("\n"):
        if current and len(current) + len(line) + 1 > max_chars:
            out.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        out.append(current)
    return out
