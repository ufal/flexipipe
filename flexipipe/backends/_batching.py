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


def join_conllu_parts(parts: List[str]) -> str:
    """The CoNLL-U results of the parts of one text (from `split_text`) as one document.

    Each part comes back numbered on its own (`# sent_id = 1` again, a new `# newdoc`).
    Pasted together as they are, sentence ids repeat, and a reader that groups tokens by
    sent_id merges sentences far apart (xmltokenizer then cut them at every paragraph in
    between and dropped their heads: ELTeC, 2026-11-26). Sentence ids are renumbered
    1, 2, 3, ... across the parts; `# newdoc` is kept only before the first sentence.
    """
    out: List[str] = []
    n = 0
    seen_doc = False
    for part in parts:
        part = part.rstrip("\n")
        if not part.strip():
            continue
        if out:
            out.append("")
        for line in part.split("\n"):
            if line.startswith("# newdoc"):
                if seen_doc:
                    continue
                seen_doc = True
            elif line.startswith("# sent_id"):
                n += 1
                line = f"# sent_id = {n}"
            out.append(line)
    return "\n".join(out) + "\n" if out else "\n"


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
