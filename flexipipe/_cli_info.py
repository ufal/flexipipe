"""CLI entry point for the info command only. Parses args after 'info' and delegates to run_info_cli."""
from __future__ import annotations

import argparse
import sys

from ._cli_shared import add_logging_args, get_parent_parser
from .backend_registry import get_backend_choices


def _build_parser() -> argparse.ArgumentParser:
    """Build only the info command parser (same arguments and subparsers as in _cli.py for 'info')."""
    parent_parser = get_parent_parser()

    # info ---------------------------------------------------------------
    info_parser = argparse.ArgumentParser(
        description="List information about available backends and models, or detect language",
        parents=[parent_parser],
        allow_abbrev=False,
    )
    add_logging_args(info_parser)
    info_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    info_parser.add_argument(
        "--detect-language",
        action="store_true",
        help="Detect language of the provided text or STDIN",
    )
    info_parser.add_argument(
        "--text",
        help="Text snippet to analyze (optional if using --input or STDIN)",
    )
    info_parser.add_argument(
        "--input",
        "-i",
        help="Path to a file whose contents should be analyzed",
    )
    info_parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum number of characters required to run detection (default: 20)",
    )
    info_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Show up to K candidate languages when --verbose is used (default: 3)",
    )
    info_subparsers = info_parser.add_subparsers(dest="info_action", required=False, help="Information to list")

    # info backends
    backends_parser = info_subparsers.add_parser(
        "backends",
        help="List all available backends",
        parents=[parent_parser],
    )
    backends_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )

    # info languages
    languages_parser = info_subparsers.add_parser(
        "languages",
        help="List all languages that have models available",
        parents=[parent_parser],
    )
    languages_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    languages_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force refresh of language mappings and unified catalog cache",
    )

    # info models
    models_parser = info_subparsers.add_parser(
        "models",
        help="List available models for a backend or language",
        parents=[parent_parser],
    )

    # info sessions
    sessions_parser = info_subparsers.add_parser(
        "sessions",
        help="List currently running training and tagging sessions",
        parents=[parent_parser],
    )
    sessions_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    sessions_parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Include completed/failed sessions in the output",
    )
    sessions_parser.add_argument(
        "--no-cleanup",
        action="store_true",
        dest="no_cleanup",
        help="Don't automatically clean up stale session files",
    )

    # info ud-tags
    ud_tags_parser = info_subparsers.add_parser(
        "ud-tags",
        help="List Universal Dependencies tags repository information",
        parents=[parent_parser],
    )
    ud_tags_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    ud_tags_parser.add_argument(
        "--category",
        choices=["upos", "feats", "misc", "document", "sentence", "all"],
        default="all",
        help="Category to display (default: all)",
    )

    # info examples
    examples_parser = info_subparsers.add_parser(
        "examples",
        help="List locally available example datasets",
        parents=[parent_parser],
    )
    examples_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    examples_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh of example metadata (re-download files if needed)",
    )
    examples_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Alias for --refresh (force re-download of example metadata)",
    )

    # info tasks
    tasks_parser = info_subparsers.add_parser(
        "tasks",
        help="List NLP tasks supported by flexipipe",
        parents=[parent_parser],
    )
    tasks_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )

    # info renderers
    renderers_parser = info_subparsers.add_parser(
        "renderers",
        help="List all available SVG renderers",
        parents=[parent_parser],
    )
    renderers_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )

    # info installation
    installation_parser = info_subparsers.add_parser(
        "installation",
        help="Show version, package location, and how flexipipe was installed (pip, editable, git)",
        parents=[parent_parser],
    )
    installation_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )

    # info teitok
    teitok_parser = info_subparsers.add_parser(
        "teitok",
        help="Display TEITOK settings.xml configuration (attribute mappings, language, etc.)",
        parents=[parent_parser],
    )
    teitok_parser.add_argument(
        "--teitok",
        action="store_true",
        help="Enable TEITOK mode: automatically load settings from ./Resources/settings.xml (or search for settings.xml). Use --settings to specify a custom path.",
    )
    teitok_parser.add_argument(
        "--settings",
        dest="teitok_settings",
        type=str,
        metavar="PATH",
        help="Path to settings.xml file. If not specified and --teitok is used, looks for tmp/cqpsettings.xml (merged shared+local) or ./Resources/settings.xml. Otherwise, searches for settings.xml in the current directory and parent directories.",
    )
    teitok_parser.add_argument(
        "--corpus",
        type=str,
        metavar="PATH",
        help="Path to TEITOK corpus directory or XML file (used to search for settings.xml)",
    )
    teitok_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    models_parser.add_argument(
        "--backend",
        choices=get_backend_choices(),
        help="Backend type (required unless --language is provided)",
    )
    models_parser.add_argument(
        "--language",
        help="Filter model listings by language name or ISO code (searches across all backends if --backend not specified)",
    )
    models_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force refresh of cached model listings",
    )
    models_parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table). Use 'json' for machine-readable output.",
    )
    # Add REST backend URL arguments for models listing
    models_parser.add_argument(
        "--endpoint-url",
        help="REST backend endpoint URL (for UDPipe, UDMorph, NameTag backends)",
    )
    models_parser.add_argument(
        "--sort",
        choices=["backend", "model", "language", "iso", "status"],
        default="backend",
        help="Sort order for models (default: backend). Options: backend, model, language, iso, status",
    )
    models_parser.add_argument(
        "--include-base-models",
        action="store_true",
        help="Include base/LLM-style Transformers models (without finetuning) when listing --backend transformers",
    )

    return info_parser


def main(argv: list[str] | None = None) -> int:
    """Parse info args (after 'info' if present) and run run_info_cli."""
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "info":
        argv = argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)
    from .info import run_info_cli
    return run_info_cli(args)
