"""Writing XML files safely: never replace a file with XML that does not parse.

Every place that writes a TEITOK / TEI file (in-place writeback above all) goes through
:func:`write_xml_file`:

- the exact bytes to be written are parsed first; XML that is not well-formed is never written to
  the target: it is saved under the temp folder (``flexipipe-rejected-…``) for inspection and
  :class:`InvalidXMLError` is raised, naming the problem, its line and that file — the input stays
  as it was;
- with `original` (the input of a writeback): the text must not have changed — TEITOK's text is
  the XML's text, the spacing between tokens included (teitok_text_check: character for character,
  or with `pretty` as TEITOK reads it); a changed text is refused the same way;
- the file it replaces is backed up first, always (teitok_backup: TEITOK's backups/ folder, one
  copy per write); a backup that cannot be made stops the write (BackupError);
- well-formed XML is written to a temporary file next to the target and renamed over it, so an
  interrupted run (killed, disk full) cannot leave half a file (in place when only the file, not its
  folder, is writable); a target that cannot be written gives a PermissionError saying who lacks
  which permission.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

try:
    from lxml import etree as _lxml_etree
except ImportError:  # pragma: no cover - lxml is the normal case
    _lxml_etree = None


class InvalidXMLError(RuntimeError):
    """The XML to be written does not parse, or changes the text; the target was left untouched."""

    def __init__(self, message: str, target: Path, rejected: Optional[Path]):
        super().__init__(message)
        self.target = target
        self.rejected = rejected


def xml_problem(data: bytes) -> Optional[str]:
    """Why `data` is not well-formed XML ("line N, column M: …"), or None when it is."""
    if _lxml_etree is not None:
        try:
            # huge_tree: large corpus files; no network, no entity expansion
            parser = _lxml_etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)
            _lxml_etree.fromstring(data, parser)
            return None
        except _lxml_etree.XMLSyntaxError as exc:
            line, col = getattr(exc, "position", (None, None)) or (None, None)
            where = f"line {line}, column {col}: " if line else ""
            return f"{where}{exc.msg if hasattr(exc, 'msg') else exc}"
    import xml.etree.ElementTree as ET

    try:
        ET.fromstring(data)
        return None
    except ET.ParseError as exc:
        line, col = getattr(exc, "position", (None, None)) or (None, None)
        where = f"line {line}, column {col}: " if line else ""
        return f"{where}{exc}"


def _save_rejected(data: bytes, target: Path) -> Optional[Path]:
    """Keep the rejected XML for inspection (temp folder); None when that fails too."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        fd, name = tempfile.mkstemp(prefix=f"flexipipe-rejected-{target.stem}-{stamp}-", suffix=target.suffix or ".xml")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return Path(name)
    except OSError:
        return None


def _refuse(target: Path, data: bytes, why: str) -> None:
    rejected = _save_rejected(data, target)
    kept = f"; the rejected XML is in {rejected}" if rejected else ""
    raise InvalidXMLError(f"refusing to write {target}: {why}; the file was left as it was{kept}", target, rejected)


def write_xml_file(
    target: Union[str, Path],
    data: Union[bytes, str],
    *,
    original: Union[str, Path, bytes, None] = None,
    pretty: bool = False,
    normalization: Optional[str] = None,
    backup: bool = True,
    expect_tokens: bool = False,
) -> None:
    """Write `data` (XML; str is written as UTF-8) to `target` if it parses — and, given the
    writeback's input as `original`, keeps its text — atomically.

    Raises InvalidXMLError (target untouched, the XML kept under the temp folder) when it does not.
    `normalization`: a Unicode normalization the run applied on purpose (NFC, …; "none" = none).
    `expect_tokens`: the run inserts tokens — a result without any <tok> is refused (a tokenizing
    engine that failed quietly must not count as a success).
    """
    target = Path(target)
    if isinstance(data, str):
        data = data.encode("utf-8")
    problem = xml_problem(data)
    if problem is not None:
        _refuse(target, data, f"the result is not well-formed XML ({problem})")
    if expect_tokens and not re.search(rb"<tok[\s>/]", data):
        _refuse(target, data, "the result has no tokens although the run was to insert them (the tokenizing step failed)")
    if original is not None:
        before = original if isinstance(original, bytes) else Path(original).read_bytes()
        from .teitok_text_check import spacing_change, text_change

        norm = normalization if normalization and str(normalization).lower() not in ("none", "") else None
        try:
            change = (spacing_change if pretty else text_change)(before, data, norm)
        except Exception as exc:  # the input itself does not parse: nothing to compare with
            change = None if xml_problem(before) else f"the text could not be compared ({exc})"
        if change:
            _refuse(target, data, f"the result would change the text: {change}")
    directory = target.parent if str(target.parent) else Path(".")
    exists = target.exists()
    dir_ok = os.access(directory, os.W_OK)
    if exists and not dir_ok and not os.access(target, os.W_OK) or not exists and not dir_ok:
        raise PermissionError(
            f"cannot write {target}: {_user()} may not change "
            + ("the file or its folder" if exists else f"the folder {directory}")
            + " (give that user write permission, e.g. the web server user for TEITOK)"
        )
    # a backup of what is replaced, always (not of --test's own temporary copy)
    if exists and backup and not _is_preview_copy(target):
        from .teitok_backup import backup_before_write

        backup_before_write(target)
    if not dir_ok:
        # the file is writable but not its folder: in place (checked and backed up, not atomic)
        with open(target, "r+b") as handle:
            handle.write(data)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(directory))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exists:  # keep the file's permissions (TEITOK: the web server must write it too)
            try:
                os.chmod(tmp, target.stat().st_mode & 0o7777)
            except OSError:
                pass
        else:
            os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _is_preview_copy(path: Path) -> bool:
    """--test / --dry-run writes into a temporary copy (flexipipe-test-*) — nothing to back up."""
    try:
        return path.name.startswith("flexipipe-test-") and Path(tempfile.gettempdir()).resolve() == path.resolve().parent
    except OSError:
        return False


def _user() -> str:
    try:
        import pwd

        return f"the user {pwd.getpwuid(os.geteuid()).pw_name}"
    except Exception:
        return "this user"


def write_xml_tree(target: Union[str, Path], tree, **check) -> None:
    """An lxml / ElementTree tree through :func:`write_xml_file` (UTF-8, with declaration; never
    lxml's pretty_print, which changes TEITOK's text — teitok_pretty does it safely). `check`:
    write_xml_file's original / pretty / normalization."""
    if _lxml_etree is not None and isinstance(tree, _lxml_etree._ElementTree):
        data = _lxml_etree.tostring(tree, encoding="utf-8", xml_declaration=True)
    else:
        import io

        buf = io.BytesIO()
        tree.write(buf, encoding="utf-8", xml_declaration=True)
        data = buf.getvalue()
    write_xml_file(target, data, **check)


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask
