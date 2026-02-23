"""Install subcommand: optional backends, wrapper script, or self-update. Loaded only for 'flexipipe install'."""
from __future__ import annotations

import argparse
import sys

from ._cli_shared import add_logging_args, get_parent_parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "install":
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        prog="flexipipe install",
        description="Install optional backend dependencies or the wrapper script",
        parents=[get_parent_parser()],
    )
    add_logging_args(parser)
    parser.add_argument(
        "backends",
        nargs="+",
        metavar="BACKEND",
        help="Backend(s) to install (e.g., spacy, udapi, all), or 'wrapper' to install the flexipipe launcher script, or 'update' to upgrade flexipipe",
    )
    parser.add_argument(
        "--path",
        dest="wrapper_path",
        metavar="DIR",
        help="For 'flexipipe install wrapper': install the script into DIR (or DIR/flexipipe). Without this, you are prompted.",
    )
    args = parser.parse_args(argv)
    from .install import run_install
    return run_install(args)
