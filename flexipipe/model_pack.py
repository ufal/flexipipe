"""A trained model made ready to publish: without the details of the machine it was trained on.

A model folder records where it was made — the manifest's ``trained_on``, a flexitag model's
``source_folders`` / ``source_files`` (the corpus folder, the temporary training files), paths among
the options — that is: user names, folder names, the layout of someone's disk. The folder itself keeps
them (the corpus's model list uses them: "trained on this corpus"); a copy for publishing does not.

Only the metadata is cleaned: the manifest (flexipipe-model.json) and the metadata object at the start
of a model file ("metadata" / "meta": model_vocab.json, model_pron.json). The model itself is never
touched — a vocabulary may hold word forms like "/home/" or "and/or/not", which are data. Paths of
this machine elsewhere (in the model's data, in files that are not JSON) are reported, to look at.

- check(folder): (cleanable, to look at) — [(file, where, value)].
- pack(folder, output): a .tar.gz of a copy with the metadata's paths reduced to their last part (a
  corpus folder to the corpus's name), and what is left to look at.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model_hub import INSTALLED_FROM

MANIFESTS = {"flexipipe-model.json"}  # metadata as a whole
HEAD_KEYS = ("metadata", "meta")  # the metadata object at the start of a large model file
SCAN_BYTES = 4 * 1048576  # how much of a file that is not JSON is looked through

# an absolute path of some machine: /Users/…, /home/…, C:\…, ~/…  (two parts at least)
_PATH = re.compile(r"^(?:/[^/\s]+/\S*|[A-Za-z]:[\\/]\S+|~[/\\]\S+|\\\\\S+)$")
_PATH_IN_TEXT = re.compile(rb"(?:/(?:Users|home|Volumes|private|var/folders|tmp|mnt|media|srv)/[^\s\"'<>]+|[A-Za-z]:\\\\?Users\\\\?[^\s\"'<>]+)")
_SKIP = {INSTALLED_FROM, ".DS_Store"}

Finding = Tuple[str, str, str]  # (file, where in it, the value)


def is_local_path(value: Any) -> bool:
    return isinstance(value, str) and len(value) < 4096 and bool(_PATH.match(value.strip()))


def public_name(value: str) -> str:
    """What is left of a path: its last part — of a corpus's xmlfiles/ folder, the corpus's name."""
    parts = [p for p in re.split(r"[\\/]+", value.strip()) if p and p != "~" and not re.match(r"^[A-Za-z]:$", p)]
    if not parts:
        return ""
    if parts[-1] == "xmlfiles" and len(parts) > 1:
        return parts[-2]
    return parts[-1]


def _scrub(value: Any, where: str, found: List[Tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {k: _scrub(v, f"{where}.{k}" if where else k, found) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, f"{where}[{i}]", found) for i, v in enumerate(value)]
    if is_local_path(value):
        found.append((where, value))
        return public_name(value)
    return value


def _head_span(path: Path) -> Optional[Tuple[str, int, int, Any]]:
    """The metadata object at the start of a large JSON file: (key, start, end, value) in characters."""
    with open(path, encoding="utf-8") as fh:
        head = fh.read(4 * 1048576)
    for key in HEAD_KEYS:
        m = re.search(r'"' + key + r'"\s*:\s*', head)
        if not m or not head[m.end():].startswith("{"):
            continue
        try:
            value, end = json.JSONDecoder().raw_decode(head, m.end())
        except ValueError:
            continue
        return key, m.end(), end, value
    return None


def _json_file(path: Path, rel: str, write: bool) -> Optional[List[Finding]]:
    """Paths in a JSON file's metadata (cleaned in place when write), then the rest looked through as
    text; None when it is not JSON with metadata (a plain text scan then)."""
    found: List[Tuple[str, str]] = []
    if path.name in MANIFESTS:
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (UnicodeDecodeError, ValueError):
            return None
        clean = _scrub(data, "", found)
        if write and found:
            path.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return [(rel, w, v) for w, v in found]
    try:
        span = _head_span(path)
    except (UnicodeDecodeError, OSError):
        return None
    if span is None:
        return None
    key, start, end, value = span
    clean = _scrub(value, key, found)
    if write and found:
        tmp = path.with_name(path.name + ".tmp")
        with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
            head = src.read(end)
            out.write(head[:start] + json.dumps(clean, ensure_ascii=False))
            shutil.copyfileobj(src, out, 1 << 20)
        tmp.replace(path)
    return [(rel, w, v) for w, v in found] + _text_file(path, rel, skip_head=len(head_bytes(path, end)))


def head_bytes(path: Path, chars: int) -> bytes:
    with open(path, encoding="utf-8") as fh:
        return fh.read(chars).encode("utf-8")


def _text_file(path: Path, rel: str, *, skip_head: int = 0) -> List[Finding]:
    with open(path, "rb") as fh:
        data = fh.read(SCAN_BYTES + skip_head)[skip_head:]
    hits = sorted({m.group(0).decode("utf-8", "replace") for m in _PATH_IN_TEXT.finditer(data)})
    return [(rel, "text", h) for h in hits[:5]]


def _files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.name not in _SKIP
                  and not p.name.startswith(".") and not p.name.endswith((".bak", ".tmp")))


def check(folder, *, write: bool = False) -> Tuple[List[Finding], List[Finding]]:
    """(cleanable, to look at): paths of this machine in the model folder — in the metadata (cleaned when
    write), and elsewhere (reported only)."""
    folder = Path(folder)
    cleanable: List[Finding] = []
    look: List[Finding] = []
    for f in _files(folder):
        rel = str(f.relative_to(folder))
        found = _json_file(f, rel, write) if f.suffix.lower() == ".json" else None
        if found is None:
            look += _text_file(f, rel)
            continue
        cleanable += [x for x in found if x[1] != "text"]
        look += [x for x in found if x[1] == "text"]
    return cleanable, look


def pack(folder, output=None, extra: Optional[Dict[str, Path]] = None) -> Tuple[Path, List[Finding], List[Finding]]:
    """A .tar.gz of the model folder with its metadata cleaned: (the package, what was cleaned, what is
    left to look at). extra: files to add {name in the package: file} (a corpus's tagset.xml)."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a model folder")
    name = folder.name
    output = Path(output) if output else folder.parent / f"{name}.tar.gz"
    work = Path(tempfile.mkdtemp(prefix="flexipipe-pack-"))
    try:
        copy = work / name
        copy.mkdir()
        for f in _files(folder):
            target = copy / f.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        for name_in, src in (extra or {}).items():
            shutil.copy2(src, copy / name_in)
        cleaned, _ = check(copy, write=True)
        left, look = check(copy)
        if left:  # the metadata is clean after cleaning, or nothing is packed
            raise RuntimeError("the metadata could not be cleaned: " + ", ".join(f"{f} {w}" for f, w, _ in left))
        with tarfile.open(output, "w:gz") as tar:
            tar.add(copy, arcname=name)
        return output, cleaned, look
    finally:
        shutil.rmtree(work, ignore_errors=True)
