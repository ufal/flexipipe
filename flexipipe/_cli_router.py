"""CLI entry router. Loads only the subcommand module for the requested task."""
from __future__ import annotations

import sys

from ._cli_shared import TASK_CHOICES


def _subcommand_from_argv(argv: list[str]) -> str | None:
    """First positional that looks like a task name."""
    for a in argv:
        if not a.startswith("-") and a in TASK_CHOICES:
            return a
    return None


def run(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Backward compatibility: old `show ...` command maps to `info ...`
    if argv and argv[0] == "show":
        argv = ["info", *argv[1:]]
    sub = _subcommand_from_argv(argv)
    if sub == "install":
        from ._cli_install import main
        return main(argv)
    if sub == "info":
        from ._cli_info import main
        return main(argv)
    # config, validate, benchmark, process, train, convert, or -h: full CLI
    from ._cli import main
    return main(argv)
