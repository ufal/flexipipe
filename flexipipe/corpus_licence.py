"""What a corpus's own files say about their licence: the TEI headers' ``<availability>`` and ``<licence>``.

A model trained on a corpus holds material taken from its texts (a flexitag model: a word list with
frequencies and contexts), so it can be published only as far as the texts' licence allows. TEITOK has no
corpus-wide licence setting; where a corpus states one, it is in the files' TEI headers. `licences_in`
reads those (only the headers: each file is read up to ``</teiHeader>``) and sums them up, for the model's
manifest — so publishing the model can show what the training data said, next to what the sender declares.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

MAX_FILES = 500  # enough to see what a corpus says; a large corpus is sampled evenly


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _header_licences(path: Path) -> Optional[Dict[str, List[str]]]:
    """{"status": [...], "licence": [...]} from one file's teiHeader; None when it has none / is not XML."""
    status: List[str] = []
    licences: List[str] = []
    seen_header = False
    try:
        for event, elem in ET.iterparse(str(path), events=("start", "end")):
            name = _local(elem.tag)
            if event == "start":
                if name == "teiHeader":
                    seen_header = True
                elif name == "text" and not seen_header:
                    return None  # no header before the text
                continue
            if name == "availability" and elem.get("status"):
                status.append(elem.get("status").strip())
            elif name == "licence":
                text = " ".join("".join(elem.itertext()).split())
                target = (elem.get("target") or "").strip()
                value = target if target and (not text or text == target) else (f"{text} ({target})" if target else text)
                if value:
                    licences.append(value[:200])
            elif name == "teiHeader":
                return {"status": status, "licence": licences}
    except (ET.ParseError, OSError):
        return None
    return {"status": status, "licence": licences} if seen_header else None


def _xml_files(root: Path) -> List[Path]:
    folder = root / "xmlfiles" if (root / "xmlfiles").is_dir() else root
    files: List[Path] = []
    for ext in ("*.xml", "*.tei", "*.teitok"):
        files.extend(folder.rglob(ext))
    return sorted(files)


def licences_in(source, *, max_files: int = MAX_FILES) -> Optional[dict]:
    """A summary of the licence statements in a TEITOK corpus folder (or a list of files), or None
    when there are no XML files: {"files": n read, "with_licence": n, "licences": {text: count},
    "availability": {status: count}}."""
    if isinstance(source, (str, Path)):
        root = Path(source)
        if not root.is_dir():
            return None
        files = _xml_files(root)
    else:
        files = [Path(p) for p in source]
    if not files:
        return None
    if len(files) > max_files:
        step = len(files) / max_files
        files = [files[int(i * step)] for i in range(max_files)]
    licences: Counter = Counter()
    status: Counter = Counter()
    read = with_licence = 0
    for f in files:
        h = _header_licences(f)
        if h is None:
            continue
        read += 1
        if h["licence"]:
            with_licence += 1
        licences.update(set(h["licence"]))
        status.update(set(h["status"]))
    if not read:
        return None
    return {"files": read, "with_licence": with_licence,
            "licences": dict(licences.most_common()), "availability": dict(status.most_common())}


def describe(summary: Optional[dict]) -> str:
    """One line for people: what the training files said."""
    if not summary:
        return "the training files have no TEI headers"
    n = summary["files"]
    parts = []
    if summary["licences"]:
        parts.append("; ".join(f"{k} ({v} of {n})" if v != n else k for k, v in summary["licences"].items()))
    else:
        parts.append(f"none of the {n} files names a licence")
    if summary["availability"]:
        parts.append("availability: " + ", ".join(f"{k} ({v})" for k, v in summary["availability"].items()))
    return "; ".join(parts)
