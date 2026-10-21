"""Progress of a long run, one line per stage, for whoever watches the output (the TEITOK job page
reads it): ``[flexipipe] step 2/5: building the vocabulary``. Printed to stdout and flushed at once,
so a log written by a background job shows it while the run goes on."""

from __future__ import annotations

import sys


class Steps:
    def __init__(self, total: int):
        self.total = max(1, total)
        self.done = 0

    def next(self, text: str) -> None:
        self.done = min(self.done + 1, self.total)
        print(f"[flexipipe] step {self.done}/{self.total}: {text}", file=sys.stdout, flush=True)
