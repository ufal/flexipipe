"""Shared CLI constants and minimal parser helpers. No heavy imports (no backends)."""
from __future__ import annotations

import argparse
from pathlib import Path

TASK_CHOICES = (
    "process",
    "train",
    "convert",
    "config",
    "info",
    "install",
    "benchmark",
    "validate",
)


def get_parent_parser() -> argparse.ArgumentParser:
    """Minimal parent parser with --verbose and --debug only (no backend choices)."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true", help="Print high-level progress messages")
    return p


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """No-op; logging args are in parent parser."""
    pass
