"""
A backup before every write (xml_safe_write calls it before it replaces a file).

TEITOK's convention, so TEITOK's backups page (backups.php: ``backups/<xmlid>-*``) lists them:
the first write of a day keeps the file as it was in ``backups/<name>-YYYYMMDD.xml`` (the state
before the day's first change, as TEITOK itself does); every further write that day keeps the state
before it in ``backups/<name>-YYYYMMDDTHHMMSS.xml`` (TEITOK reads that as a date too). So every
state a write replaces can be recovered.

The ``backups/`` folder is the TEITOK project's (the folder that holds ``xmlfiles/``, else the one
with ``Resources/settings.xml``); outside a project, a ``backups/`` folder next to the file.
A backup that cannot be made raises BackupError: the write must not happen without it.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class BackupError(RuntimeError):
    """The backup before a write could not be made; the write was not done."""


def project_root(path: Path) -> Optional[Path]:
    """The TEITOK project a file belongs to: the parent of its (topmost) xmlfiles/ folder, else the
    nearest folder with Resources/settings.xml; None outside a project."""
    path = path.resolve()
    parts = path.parts
    if "xmlfiles" in parts[:-1]:
        i = parts.index("xmlfiles")
        return Path(*parts[:i]) if i else Path("/")
    for folder in path.parents:
        if (folder / "Resources" / "settings.xml").is_file():
            return folder
    return None


def backups_folder(path: Path) -> Path:
    root = project_root(path)
    return (root if root is not None else path.resolve().parent) / "backups"


def backup_before_write(file_path, now: Optional[datetime] = None) -> Optional[Path]:
    """Copy `file_path` into the backups folder before it is overwritten; the backup's path, or
    None when there is no file yet. Raises BackupError when the copy cannot be made."""
    source = Path(file_path)
    if not source.is_file():
        return None
    now = now or datetime.now()
    folder = backups_folder(source)
    stem, suffix = (source.stem, source.suffix) if source.suffix else (source.name, "")
    daily = folder / f"{stem}-{now:%Y%m%d}{suffix}"
    target = daily if not daily.exists() else folder / f"{stem}-{now:%Y%m%dT%H%M%S}{suffix}"
    n = 1
    while target.exists():  # two writes within one second
        target = folder / f"{stem}-{now:%Y%m%dT%H%M%S}-{n}{suffix}"
        n += 1
    try:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    except OSError as exc:
        raise BackupError(f"the backup of {source} could not be made in {folder} ({exc})") from exc
    return target


def create_teitok_backup(file_path: str, execution_dir: Optional[Path] = None) -> Optional[Path]:
    """Older entry point (kept for callers): the same backup, None when it cannot be made."""
    try:
        return backup_before_write(file_path)
    except BackupError:
        return None
