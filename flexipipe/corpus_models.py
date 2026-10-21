"""Models trained on a corpus, kept with it: ``Resources/flexipipe-models/<date>/`` in a TEITOK
project (dev/TEITOK-TRAINING.md). They belong to the corpus, the web server can write there, and
they do not depend on any user's flexipipe configuration.

The one in use is settings.xml ``<flexipipe><training use="Resources/flexipipe-models/…"/>``, else
the newest. It is the default for the steps it provides that the corpus's main model does not (a
pronunciation model: task ``pron``; task_routing), and it can be the corpus's main model (its folder
as the flexitag model).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

FOLDER = "Resources/flexipipe-models"


def corpus_root(settings_path) -> Optional[Path]:
    """The project folder of a settings file (Resources/settings.xml, tmp/cqpsettings.xml)."""
    if not settings_path:
        return None
    return Path(settings_path).resolve().parent.parent


def model_folders(root) -> List[Path]:
    """The corpus's models (folders with a flexipipe-model.json, or a flexitag model_vocab.json), newest
    first (folders are named by date)."""
    if root is None:
        return []
    base = Path(root) / FOLDER
    try:
        folders = {p.parent for pattern in ("*/flexipipe-model.json", "*/model_vocab.json") for p in base.glob(pattern)}
    except OSError:
        return []
    return sorted(folders, key=lambda p: p.name, reverse=True)


def model_to_load(folder: Path) -> tuple:
    """(backend, model path) to run a corpus model: its flexipipe-model.json, else flexitag on the folder."""
    import json

    try:
        data = json.loads((Path(folder) / "flexipipe-model.json").read_text(encoding="utf-8"))
        model = Path(data.get("model") or ".")
        return data.get("backend") or "flexitag", str(model if model.is_absolute() else Path(folder) / model)
    except (OSError, ValueError):
        return "flexitag", str(folder)


def in_use(root, use: Optional[str] = None) -> Optional[Path]:
    """The model in use: `use` (settings: <training use>, relative to the project) when it is there,
    else the newest."""
    folders = model_folders(root)
    if use:
        p = Path(use)
        p = p if p.is_absolute() else Path(root) / p
        for f in folders:
            if f.resolve() == p.resolve():
                return f
    return folders[0] if folders else None


def list_models(root, use: Optional[str] = None) -> List[dict]:
    """Every corpus model as a model card (model_card.py) + ``folder`` (relative to the project),
    ``about`` and ``in_use``."""
    from .model_card import describe, model_card

    current = in_use(root, use)
    out = []
    for f in model_folders(root):
        card = model_card(f, root)
        card["folder"] = str(f.relative_to(Path(root)))
        card["about"] = describe(card)
        backend, model = model_to_load(f)
        card["backend"] = backend
        try:  # relative to the project, as settings.xml names it
            card["load"] = str(Path(model).resolve().relative_to(Path(root).resolve()))
        except ValueError:
            card["load"] = model
        card["in_use"] = current is not None and f == current
        out.append(card)
    return out
