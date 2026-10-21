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
        help="Backend(s) to install (e.g., spacy, udapi, all), 'model <backend>/<model>' to install a model from the registry (or 'model FILE.tar.gz' from a package), 'wrapper' to install the flexipipe launcher script, or 'update' to upgrade flexipipe",
    )
    parser.add_argument(
        "--path",
        dest="wrapper_path",
        metavar="DIR",
        help="For 'flexipipe install wrapper': install the script into DIR (or DIR/flexipipe). Without this, you are prompted.",
    )
    parser.add_argument("--name", dest="model_name_as", metavar="NAME",
                        help="For 'flexipipe install model FILE': the name to install the package's model under")
    parser.add_argument("--backend", dest="model_backend", metavar="BACKEND",
                        help="For 'flexipipe install model FILE': the backend, when the package does not say")
    parser.add_argument("--force", action="store_true",
                        help="For 'flexipipe install model': replace a model that is already installed")
    args = parser.parse_args(argv)
    from .install import run_install
    return run_install(args)
