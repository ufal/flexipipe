"""Punctuation splitting for UDPipe: pre-split plaintext (preferred) or post-split tokens.

Modes (increasingly aggressive) via ``--option punctuation-split:MODE``:

- ``boundary`` / ``quotes`` — only punctuation that should not appear glued to
  words in well-trained models (guillemets, quote marks, etc.).
- ``hard`` (default when enabled) — split any glued punctuation except marks
  that may legitimately occur inside a token: ``. - ' ? ;`` in abbreviations,
  hyphenations, possessives (``A.D.S.L.``, ``Mr.``, ``John's``, ``semi-colon``).
- ``harder`` — like ``hard``, but also split ``.`` unless it matches those
  internal patterns (helps OCR glued ``word.»`` without forcing ``Mr .``).
- ``full`` — split every punctuation category at a letter/digit boundary
  (Czech-style ``Mr . Johnson``); use only when the model expects it.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from .doc import Document, Token

if TYPE_CHECKING:
    from .teitok_writeback_xt import XmltokenizerSession

# Shared with TEITOK xmltokenize-style peeling (insert_tokens_teitok._is_punctuation).
_EXTRA_PUNCT = ".,;:!?\"'„‟‚‛″′‹›«»()[]{}…-–—"

# Punctuation UDPipe usually never glues to alphanumeric stems (training gaps).
_BOUNDARY_ONLY_PUNCT = frozenset("«»„‹›""''‚‛″′()[]{}…–—")

# May stay inside a token at ``hard`` (not split at letter↔char boundary).
_TOKEN_INTERNAL_PUNCT = frozenset(".-'?;")

PUNCTUATION_SPLIT_MODES = ("off", "boundary", "quotes", "hard", "harder", "full")
_MODE_ALIASES = {
    "quote": "quotes",
    "boundaries": "boundary",
    "maximum": "full",
    "max": "full",
}


@dataclass(frozen=True)
class PunctuationSplitConfig:
    mode: str

    @classmethod
    def from_mode(cls, mode: Optional[str]) -> Optional["PunctuationSplitConfig"]:
        if not mode or mode == "off":
            return None
        normalized = _MODE_ALIASES.get(mode, mode)
        if normalized not in PUNCTUATION_SPLIT_MODES:
            raise ValueError(
                f"unknown punctuation-split mode {mode!r}; "
                f"use one of: {', '.join(m for m in PUNCTUATION_SPLIT_MODES if m != 'off')}"
            )
        return cls(normalized)

    def active(self) -> bool:
        return self.mode not in ("off",)


def is_punctuation_char(ch: str) -> bool:
    if not ch or ch.isspace():
        return False
    if unicodedata.category(ch).startswith("P"):
        return True
    return ch in _EXTRA_PUNCT


def _is_word_char(ch: str) -> bool:
    if not ch:
        return False
    cat = unicodedata.category(ch)
    return cat.startswith(("L", "N"))


def _apostrophe_inside_word(prev: str, ch: str, nxt: str) -> bool:
    if ch not in ("'", "'", "ʼ", "ʻ"):
        return False
    return _is_word_char(prev) and _is_word_char(nxt)


def _hyphen_inside_word(prev: str, ch: str, nxt: str) -> bool:
    if ch not in ("-", "‐", "‑", "–", "—"):
        return False
    return _is_word_char(prev) and _is_word_char(nxt)


def _semicolon_inside_word(prev: str, ch: str, nxt: str) -> bool:
    if ch != ";":
        return False
    return _is_word_char(prev) and _is_word_char(nxt)


def _question_inside_word(prev: str, ch: str, nxt: str) -> bool:
    if ch != "?":
        return False
    return _is_word_char(prev) and _is_word_char(nxt)


def _period_may_stay_internal(text: str, index: int) -> bool:
    """True when ``.`` at *index* is plausibly part of one token (``Mr.``, ``A.D.``)."""
    if index <= 0 or index >= len(text) or text[index] != ".":
        return False
    prev = text[index - 1]
    if not _is_word_char(prev):
        return False
    nxt = text[index + 1] if index + 1 < len(text) else ""
    if _is_word_char(nxt):
        return True
    if nxt.isspace():
        j = index + 1
        while j < len(text) and text[j].isspace():
            j += 1
        return j < len(text) and _is_word_char(text[j])
    return False


def _char_may_stay_inside_token(
    ch: str, prev: str, nxt: str, text: str, index: int, mode: str
) -> bool:
    if _apostrophe_inside_word(prev, ch, nxt):
        return True
    if _hyphen_inside_word(prev, ch, nxt):
        return True
    if _semicolon_inside_word(prev, ch, nxt):
        return True
    if _question_inside_word(prev, ch, nxt):
        return True
    if ch == ".":
        if mode == "hard":
            return True
        if mode == "harder":
            return _period_may_stay_internal(text, index)
    return False


def _needs_space_between(
    prev: str, ch: str, nxt: str, *, text: str, index: int, config: PunctuationSplitConfig
) -> bool:
    """Whether to insert a space between non-space *prev* and *ch*."""
    mode = config.mode
    if mode in ("off",):
        return False
    if mode in ("boundary", "quotes"):
        if ch in _BOUNDARY_ONLY_PUNCT and _is_word_char(prev):
            return True
        if prev in _BOUNDARY_ONLY_PUNCT and _is_word_char(ch):
            return True
        return False
    if is_punctuation_char(ch) and _is_word_char(prev):
        if mode == "full":
            return True
        if _char_may_stay_inside_token(ch, prev, nxt, text, index, mode):
            return False
        return True
    if is_punctuation_char(prev) and _is_word_char(ch):
        if mode == "full":
            return True
        prev_prev = text[index - 2] if index >= 2 else ""
        if _char_may_stay_inside_token(prev, prev_prev, ch, text, index - 1, mode):
            return False
        return True
    return False


def separate_punctuation_in_plaintext(
    text: str, config: Optional[PunctuationSplitConfig] = None
) -> str:
    """Insert spaces between letters and punctuation that should not be glued."""
    if not text:
        return text
    cfg = config or PunctuationSplitConfig("hard")
    if not cfg.active():
        return text
    out: List[str] = []
    for i, ch in enumerate(text):
        if i > 0:
            prev = text[i - 1]
            if not prev.isspace() and not ch.isspace():
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if _needs_space_between(prev, ch, nxt, text=text, index=i, config=cfg):
                    out.append(" ")
        out.append(ch)
    return "".join(out)


def _splittable_for_peel(ch: str, config: PunctuationSplitConfig) -> bool:
    if config.mode in ("boundary", "quotes"):
        return ch in _BOUNDARY_ONLY_PUNCT
    if config.mode == "full":
        return is_punctuation_char(ch)
    if ch == ".":
        return config.mode == "harder"  # hard: never peel '.' from edges
    if ch in _TOKEN_INTERNAL_PUNCT:
        return False
    return is_punctuation_char(ch)


def peel_edge_punctuation(
    form: str, config: Optional[PunctuationSplitConfig] = None
) -> Tuple[str, str, str]:
    """Return ``(leading, core, trailing)`` punctuation runs peeled from *form*."""
    cfg = config or PunctuationSplitConfig("hard")
    if not cfg.active():
        return "", form, ""

    def peel_leading(s: str) -> Tuple[str, str]:
        out: List[str] = []
        i = 0
        while i < len(s) and _splittable_for_peel(s[i], cfg):
            if s[i] in ("'", "'", "ʼ", "ʻ") and i + 1 < len(s):
                nxt = unicodedata.category(s[i + 1])
                if nxt.startswith("L") or nxt.startswith("N"):
                    break
            out.append(s[i])
            i += 1
        return "".join(out), s[i:]

    def peel_trailing(s: str) -> Tuple[str, str]:
        out: List[str] = []
        j = len(s) - 1
        while j >= 0 and _splittable_for_peel(s[j], cfg):
            if s[j] in ("'", "'", "ʼ", "ʻ") and j > 0:
                prev = unicodedata.category(s[j - 1])
                if prev.startswith("L") or prev.startswith("N"):
                    break
            out.insert(0, s[j])
            j -= 1
        return s[: j + 1], "".join(out)

    leading, rest = peel_leading(form)
    core, trailing = peel_trailing(rest)
    return leading, core, trailing


def maybe_prepare_nlp_plaintext_for_backend(
    raw_text: str,
    *,
    parsed: Optional[dict[str, dict[str, Any]]],
    xt_session: Optional["XmltokenizerSession"] = None,
) -> Tuple[str, bool]:
    """Return ``(text_for_udpipe, pre_split_applied)``."""
    del xt_session
    cfg = punctuation_split_config_from_options(parsed)
    if cfg is None:
        return raw_text, False
    return separate_punctuation_in_plaintext(raw_text, cfg), True


def sync_space_after_to_surface(document: Document, surface_nlp: str) -> None:
    """Set ``space_after=False`` where two consecutive token forms touch in *surface_nlp*."""
    if not surface_nlp:
        return
    cursor = 0
    for sent in document.sentences:
        tokens = sent.tokens
        for idx in range(len(tokens) - 1):
            left = tokens[idx]
            right = tokens[idx + 1]
            a, b = left.form or "", right.form or ""
            if not a or not b:
                continue
            pos = surface_nlp.find(a, cursor)
            if pos < 0:
                pos = surface_nlp.find(a)
            if pos < 0:
                continue
            end_a = pos + len(a)
            if end_a + len(b) <= len(surface_nlp) and surface_nlp[end_a : end_a + len(b)] == b:
                left.space_after = False
                cursor = end_a
            else:
                cursor = max(cursor, end_a)


def _punct_token(
    form: str,
    *,
    head: int,
    space_after: Optional[bool],
    template: Token,
) -> Token:
    tok = copy.copy(template)
    tok.form = form
    tok.lemma = form
    tok.upos = "PUNCT"
    if not tok.xpos or tok.xpos.startswith("F%"):
        tok.xpos = "Z:-------------"
    tok.feats = "_"
    tok.deprel = "punct"
    tok.head = head
    tok.deps = ""
    tok.misc = re.sub(r"TokenRange=\d+:\d+", "TokenRange=_", tok.misc or "")
    tok.space_after = space_after
    tok.is_mwt = False
    tok.mwt_start = 0
    tok.mwt_end = 0
    tok.parts = []
    tok.subtokens = []
    tok.tokid = ""
    tok.reg = ""
    tok.expan = ""
    return tok


def _core_token_from_split(
    tok: Token,
    core: str,
    *,
    space_after: Optional[bool],
) -> Token:
    out = copy.copy(tok)
    out.form = core
    if out.lemma and out.lemma != tok.form:
        pass
    elif out.lemma == tok.form:
        out.lemma = core
    out.misc = re.sub(r"TokenRange=\d+:\d+", "TokenRange=_", out.misc or "")
    out.space_after = space_after
    return out


def split_sentence_tokens(
    tokens: List[Token], config: Optional[PunctuationSplitConfig] = None
) -> List[Token]:
    """Split tokens with leading/trailing punctuation into separate PUNCT tokens."""
    cfg = config or PunctuationSplitConfig("hard")
    if not tokens or not cfg.active():
        return tokens

    expanded: List[Token] = []
    old_to_core_index: dict[int, int] = {}

    for tok in tokens:
        old_id = tok.id
        form = tok.form or ""
        if tok.is_mwt or tok.subtokens or not form:
            old_to_core_index[old_id] = len(expanded)
            expanded.append(tok)
            continue

        leading, core, trailing = peel_edge_punctuation(form, cfg)
        if not core or (not leading and not trailing):
            old_to_core_index[old_id] = len(expanded)
            expanded.append(tok)
            continue

        chunk: List[Token] = []
        if leading:
            lead_tok = _punct_token(
                leading,
                head=0,
                space_after=False,
                template=tok,
            )
            lead_tok.attrs["_punct_split_side"] = "leading"
            chunk.append(lead_tok)
        core_tok = _core_token_from_split(
            tok,
            core,
            space_after=False if trailing else tok.space_after,
        )
        chunk.append(core_tok)
        if trailing:
            trail_tok = _punct_token(
                trailing,
                head=0,
                space_after=tok.space_after,
                template=tok,
            )
            trail_tok.attrs["_punct_split_side"] = "trailing"
            chunk.append(trail_tok)

        core_idx = 1 if leading else 0
        old_to_core_index[old_id] = len(expanded) + core_idx
        expanded.extend(chunk)

    for idx, tok in enumerate(expanded, start=1):
        tok.id = idx

    id_map = {
        old: expanded[core_index].id
        for old, core_index in old_to_core_index.items()
    }
    for tok in expanded:
        if tok.head == 0:
            continue
        tok.head = id_map.get(tok.head, tok.head)

    for i, tok in enumerate(expanded):
        side = tok.attrs.pop("_punct_split_side", None)
        if side == "leading" and i + 1 < len(expanded):
            tok.head = expanded[i + 1].id
        elif side == "trailing" and i > 0:
            tok.head = expanded[i - 1].id

    return expanded


def split_sentence_tokens_hard(tokens: List[Token]) -> List[Token]:
    """Backward-compatible alias for post-split at ``hard``."""
    return split_sentence_tokens(tokens, PunctuationSplitConfig("hard"))


def apply_punctuation_split(
    document: Document, config: PunctuationSplitConfig, *, phase: str
) -> Document:
    for sent in document.sentences:
        sent.tokens = split_sentence_tokens(sent.tokens, config)
    document.meta["_punctuation_split"] = phase
    document.meta["_punctuation_split_mode"] = config.mode
    return document


def apply_punctuation_split_hard(document: Document) -> Document:
    """Backward-compatible post-UDPipe fallback at ``hard``."""
    return apply_punctuation_split(document, PunctuationSplitConfig("hard"), phase="post")


def token_has_glued_punctuation(
    tok: Token, config: Optional[PunctuationSplitConfig] = None
) -> bool:
    cfg = config or PunctuationSplitConfig("hard")
    if tok.is_mwt or tok.subtokens or not tok.form:
        return False
    _lead, core, _trail = peel_edge_punctuation(tok.form, cfg)
    return bool(core) and tok.form != core


def punctuation_split_config_from_options(
    parsed: Optional[dict[str, dict[str, Any]]],
) -> Optional[PunctuationSplitConfig]:
    """Parse ``punctuation-split`` namespace from ``--option`` values."""
    if not parsed:
        return None
    ps = parsed.get("punctuation-split", {})
    if not ps:
        return None

    if ps.get("off") is True or ps.get("false") is True or ps.get("0") is True:
        return None

    explicit = ps.get("mode")
    if isinstance(explicit, str) and explicit.strip():
        return PunctuationSplitConfig.from_mode(explicit.strip().lower())

    # Flag form: punctuation-split:hard → {"hard": True}; pick strongest if several.
    order = ("full", "harder", "hard", "boundary", "quotes")
    for name in order:
        if ps.get(name) is True:
            return PunctuationSplitConfig.from_mode(name)
    for key, val in ps.items():
        if val is True and key in PUNCTUATION_SPLIT_MODES:
            return PunctuationSplitConfig.from_mode(key)
    return None


def punctuation_split_mode_from_options(
    parsed: Optional[dict[str, dict[str, Any]]],
) -> Optional[str]:
    """Return mode name or ``None`` if disabled."""
    cfg = punctuation_split_config_from_options(parsed)
    return cfg.mode if cfg else None


def apply_punctuation_split_from_options(
    document: Document,
    parsed: Optional[dict[str, dict[str, Any]]],
) -> Document:
    """Post-tag fallback, or ``SpaceAfter=No`` repair after pre-split."""
    cfg = punctuation_split_config_from_options(parsed)
    if document.meta.get("_punctuation_split") == "pre":
        surface = document.meta.get("_teitok_extracted_nlp") or ""
        sync_space_after_to_surface(document, surface)
        return document
    if cfg is None:
        return document
    if not any(
        token_has_glued_punctuation(tok, cfg)
        for sent in document.sentences
        for tok in sent.tokens
    ):
        return document
    return apply_punctuation_split(document, cfg, phase="post")
