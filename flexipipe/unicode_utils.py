from __future__ import annotations

import unicodedata
from typing import Literal

UnicodeForm = Literal["none", "NFC", "NFD"]


def normalize_unicode(text: str | None, form: UnicodeForm = "none") -> str | None:
    """Normalize text to the given Unicode form.

    form:
      - "none": return the text unchanged
      - "NFC": canonical composition
      - "NFD": canonical decomposition
    """
    if text is None or form == "none":
        return text
    if form not in {"NFC", "NFD"}:
        raise ValueError(f"Unsupported unicode normalization form: {form}")
    return unicodedata.normalize(form, text)
