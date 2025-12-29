"""Built-in language detectors."""

from __future__ import annotations

# Import built-in detectors to register them
from . import fasttext_detector as _fasttext_detector  # noqa: F401
from . import trigram_detector as _trigram_detector  # noqa: F401

# Import optional detectors (will fail gracefully if not installed)
try:
    from . import langdetect_detector as _langdetect_detector  # noqa: F401
except ImportError:
    pass

try:
    from . import langid_detector as _langid_detector  # noqa: F401
except ImportError:
    pass

try:
    from . import heliport_detector as _heliport_detector  # noqa: F401
except ImportError:
    pass

