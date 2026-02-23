"""Entry point for python -m flexipipe and console_scripts. Kept minimal so --version is fast."""
from __future__ import annotations

import sys

# Single source for --version output; release script updates this and __init__.py.
_VERSION = "0.3.6"

# Handle --version / -V before any other imports (no I/O, no heavy modules).
if "--version" in sys.argv or "-V" in sys.argv:
    print(f"flexipipe {_VERSION}")
    sys.exit(0)

from ._cli_router import run

if __name__ == "__main__":
    raise SystemExit(run())
