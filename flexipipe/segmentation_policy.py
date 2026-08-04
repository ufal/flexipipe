"""
Segmentation policy for TEITOK documents without <s> elements.

Flexitag/neotag only needs a token stream per Viterbi lattice; sentence boundaries
control start_transitions and lattice length. This module applies explicit policies
instead of relying on accidental XML block structure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .doc import Document, Sentence, Token

# Freeling-style sentence-final and closing punctuation tags
DEFAULT_PUNCT_XPOS = frozenset({"Fp", "Fit", "Fat", "Fe"})
DEFAULT_CLOSING_PUNCT_XPOS = frozenset({"Frt", "Faa", "Fat", "Fit"})
VALID_MODES = frozenset({"native", "document", "block", "heuristic", "model"})
SENTENCE_END_FORMS = frozenset({".", "!", "?", "…", "。", "！", "？"})


@dataclass
class HeuristicConfig:
    split_on: Tuple[str, ...] = ("block", "lb", "punct")
    punct_xpos: Tuple[str, ...] = tuple(sorted(DEFAULT_PUNCT_XPOS))
    closing_punct_xpos: Tuple[str, ...] = tuple(sorted(DEFAULT_CLOSING_PUNCT_XPOS))
    max_tokens: Optional[int] = 300

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HeuristicConfig":
        if not data:
            return cls()
        split_on = tuple(data.get("split_on") or ("block", "lb", "punct"))
        punct_xpos = tuple(data.get("punct_xpos") or sorted(DEFAULT_PUNCT_XPOS))
        closing = tuple(data.get("closing_punct_xpos") or sorted(DEFAULT_CLOSING_PUNCT_XPOS))
        max_tokens = data.get("max_tokens")
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        return cls(
            split_on=split_on,
            punct_xpos=punct_xpos,
            closing_punct_xpos=closing,
            max_tokens=max_tokens,
        )


@dataclass
class ModelSegmenterConfig:
    backend: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    fallback: Tuple[str, ...] = ("udpipe", "sentencepiece")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ModelSegmenterConfig":
        if not data:
            return cls()
        fallback = tuple(data.get("fallback") or ("udpipe", "sentencepiece"))
        return cls(
            backend=data.get("backend"),
            model=data.get("model"),
            language=data.get("language"),
            fallback=fallback,
        )


@dataclass
class SegmentationPolicy:
    mode: str = "block"
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    model: ModelSegmenterConfig = field(default_factory=ModelSegmenterConfig)

    def __post_init__(self) -> None:
        self.mode = (self.mode or "block").lower()
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid segmentation mode '{self.mode}'. "
                f"Choose from: {', '.join(sorted(VALID_MODES))}"
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SegmentationPolicy":
        if not data:
            return cls()
        if isinstance(data, str):
            return cls(mode=data)
        mode = data.get("mode", "block")
        return cls(
            mode=mode,
            heuristic=HeuristicConfig.from_dict(data.get("heuristic")),
            model=ModelSegmenterConfig.from_dict(data.get("model")),
        )


@dataclass
class TokenSegmentContext:
    """Per-token XML context used for heuristic segmentation."""

    block_key: str = ""
    line_key: str = ""


def default_policy_for_target(target: str) -> SegmentationPolicy:
    """Return backend-appropriate default segmentation policy."""
    target_key = (target or "").lower()
    if target_key in ("flexitag", "flexitag_train"):
        return SegmentationPolicy(mode="document")
    if target_key in ("conllu", "neural", "train", "convert"):
        return SegmentationPolicy(mode="heuristic")
    return SegmentationPolicy(mode="block")


def _is_sentence_final_token(token: Token, punct_xpos: frozenset[str]) -> bool:
    xpos = (token.xpos or "").strip()
    if xpos in punct_xpos:
        return True
    form = (token.form or "").strip()
    if form in SENTENCE_END_FORMS:
        return True
    if len(form) == 1 and form in ".!?":
        return True
    return False


def _is_closing_punct_token(token: Token, closing_xpos: frozenset[str]) -> bool:
    return (token.xpos or "").strip() in closing_xpos


def _finalize_sentence(tokens: List[Token], sent_counter: int) -> Sentence:
    if tokens:
        tokens[-1].space_after = None
    parts: List[str] = []
    for tok in tokens:
        parts.append(tok.form)
        if tok.space_after:
            parts.append(" ")
    sent_id = f"s-{sent_counter}"
    return Sentence(
        id=sent_id,
        sent_id=sent_id,
        source_id=sent_id,
        text="".join(parts).strip(),
        tokens=list(tokens),
        attrs={},
    )


def _renumber_tokens(sentences: List[Sentence]) -> None:
    next_id = 1
    for sentence in sentences:
        for token in sentence.tokens:
            if token.is_mwt and token.subtokens:
                token.id = next_id
                token.mwt_start = next_id
                token.mwt_end = next_id + len(token.subtokens) - 1
                for i, sub in enumerate(token.subtokens, start=1):
                    sub.id = i
                next_id += len(token.subtokens)
            else:
                token.id = next_id
                next_id += 1


def split_tokens_heuristic(
    tokens: Sequence[Token],
    contexts: Sequence[TokenSegmentContext],
    config: HeuristicConfig,
) -> List[Sentence]:
    if not tokens:
        return []

    split_on = {s.lower() for s in config.split_on}
    punct_xpos = frozenset(config.punct_xpos)
    closing_xpos = frozenset(config.closing_punct_xpos)
    max_tokens = config.max_tokens

    sentences: List[Sentence] = []
    current: List[Token] = []
    sent_counter = 1
    prev_block = ""
    prev_line = ""
    pending_close = False

    def flush() -> None:
        nonlocal sent_counter, pending_close
        if not current:
            pending_close = False
            return
        sentences.append(_finalize_sentence(current, sent_counter))
        sent_counter += 1
        current.clear()
        pending_close = False

    for idx, token in enumerate(tokens):
        ctx = contexts[idx] if idx < len(contexts) else TokenSegmentContext()
        block_key = ctx.block_key or ""
        line_key = ctx.line_key or ""

        if current:
            if "block" in split_on and block_key != prev_block:
                flush()
            elif "lb" in split_on and line_key != prev_line and prev_line:
                flush()

        current.append(token)

        if max_tokens and len(current) >= max_tokens:
            flush()
            prev_block = block_key
            prev_line = line_key
            continue

        if pending_close:
            if _is_closing_punct_token(token, closing_xpos):
                prev_block = block_key
                prev_line = line_key
                continue
            flush()

        if "punct" in split_on and _is_sentence_final_token(token, punct_xpos):
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            if nxt and _is_closing_punct_token(nxt, closing_xpos):
                pending_close = True
            else:
                flush()

        prev_block = block_key
        prev_line = line_key

    if current:
        sentences.append(_finalize_sentence(current, sent_counter))

    return sentences


def split_tokens_block(
    tokens: Sequence[Token],
    contexts: Sequence[TokenSegmentContext],
) -> List[Sentence]:
    if not tokens:
        return []

    sentences: List[Sentence] = []
    current: List[Token] = []
    sent_counter = 1
    prev_block = ""

    for idx, token in enumerate(tokens):
        ctx = contexts[idx] if idx < len(contexts) else TokenSegmentContext()
        block_key = ctx.block_key or ""
        if current and block_key != prev_block:
            sentences.append(_finalize_sentence(current, sent_counter))
            sent_counter += 1
            current = []
        current.append(token)
        prev_block = block_key

    if current:
        sentences.append(_finalize_sentence(current, sent_counter))

    return sentences


def split_tokens_document(tokens: Sequence[Token]) -> List[Sentence]:
    if not tokens:
        return []
    return [_finalize_sentence(list(tokens), 1)]


def _distribute_tokens_by_text(tokens: Sequence[Token], sentence_texts: Sequence[str]) -> List[List[Token]]:
    """Assign tokens to sentences by greedy text matching."""
    if not sentence_texts:
        return [list(tokens)]
    if len(sentence_texts) == 1:
        return [list(tokens)]

    groups: List[List[Token]] = [[] for _ in sentence_texts]
    sent_idx = 0
    built = ""

    def norm(s: str) -> str:
        return " ".join(s.split())

    for token in tokens:
        if sent_idx >= len(sentence_texts):
            groups[-1].append(token)
            continue
        groups[sent_idx].append(token)
        piece = token.form + (" " if token.space_after else "")
        built += piece
        target = norm(sentence_texts[sent_idx])
        if norm(built) == target or (target and norm(built).endswith(target)):
            sent_idx += 1
            built = ""

    return groups


def split_tokens_model(
    tokens: Sequence[Token],
    config: ModelSegmenterConfig,
    *,
    language: Optional[str] = None,
    verbose: bool = False,
) -> List[Sentence]:
    if not tokens:
        return []

    full_text_parts: List[str] = []
    for token in tokens:
        full_text_parts.append(token.form)
        if token.space_after:
            full_text_parts.append(" ")
    full_text = "".join(full_text_parts).strip()

    source_doc = Document(id="segmentation-source")
    source_doc.sentences = [Sentence(id="s-1", text=full_text, tokens=[])]

    segmenter: Optional[Tuple[str, Optional[str]]] = None
    if config.backend:
        from .segmentation import parse_segmenter_spec

        spec = f"{config.backend}:{config.model}" if config.model else config.backend
        segmenter = parse_segmenter_spec(spec)
    else:
        from .segmentation import get_default_segmenter_for_language

        segmenter = get_default_segmenter_for_language(
            language or config.language,
            fallback_to_flexitag=False,
            fallback_to_sentencepiece=True,
        )
        if not segmenter:
            for fb in config.fallback:
                if fb == "sentencepiece":
                    segmenter = ("sentencepiece", None)
                    break
                segmenter = get_default_segmenter_for_language(
                    language or config.language,
                    fallback_to_flexitag=(fb == "flexitag"),
                    fallback_to_sentencepiece=(fb == "sentencepiece"),
                )
                if segmenter:
                    break

    if not segmenter:
        from .unicode_tokenizer import segment_sentences

        sentence_texts = segment_sentences(full_text, preserve_quotes=True)
    else:
        try:
            from .segmentation import apply_segmentation

            segmented = apply_segmentation(
                source_doc,
                segmenter,
                language=language or config.language,
                verbose=verbose,
                pretokenize=False,
            )
            sentence_texts = [s.text for s in segmented.sentences if s.text and s.text.strip()]
        except Exception:
            from .unicode_tokenizer import segment_sentences

            sentence_texts = segment_sentences(full_text, preserve_quotes=True)

    if not sentence_texts:
        return split_tokens_document(tokens)

    token_groups = _distribute_tokens_by_text(tokens, sentence_texts)
    sentences: List[Sentence] = []
    for i, group in enumerate(token_groups, start=1):
        if not group:
            continue
        sentences.append(_finalize_sentence(group, i))
    return sentences


def build_sentences_from_tokens(
    tokens: Sequence[Token],
    contexts: Sequence[TokenSegmentContext],
    policy: SegmentationPolicy,
    *,
    language: Optional[str] = None,
    verbose: bool = False,
) -> List[Sentence]:
    mode = policy.mode
    if mode == "document":
        return split_tokens_document(tokens)
    if mode == "block":
        return split_tokens_block(tokens, contexts)
    if mode == "heuristic":
        return split_tokens_heuristic(tokens, contexts, policy.heuristic)
    if mode == "model":
        return split_tokens_model(tokens, policy.model, language=language, verbose=verbose)
    raise ValueError(f"Unsupported segmentation mode for token building: {mode}")


def merge_document_to_single_sentence(document: Document) -> Document:
    """Merge all sentences in a document into one (document-level lattice)."""
    all_tokens: List[Token] = []
    for sentence in document.sentences:
        all_tokens.extend(copy.deepcopy(sentence.tokens))
    if not all_tokens:
        return document
    new_doc = copy.deepcopy(document)
    new_doc.sentences = split_tokens_document(all_tokens)
    _renumber_tokens(new_doc.sentences)
    return new_doc


def apply_policy(
    document: Document,
    policy: SegmentationPolicy,
    *,
    token_contexts: Optional[Sequence[TokenSegmentContext]] = None,
    language: Optional[str] = None,
    verbose: bool = False,
    force: bool = False,
) -> Document:
    """
    Apply segmentation policy to a loaded document.

    If the document has native <s> elements and mode is native, leave unchanged
    unless force=True.
    """
    has_s = bool(document.meta.get("_teitok_has_s_elements"))
    mode = policy.mode

    if has_s and mode == "native" and not force:
        return document

    all_tokens: List[Token] = []
    for sentence in document.sentences:
        all_tokens.extend(copy.deepcopy(sentence.tokens))

    if not all_tokens:
        return document

    contexts: List[TokenSegmentContext]
    if token_contexts is not None and len(token_contexts) == len(all_tokens):
        contexts = list(token_contexts)
    else:
        contexts = [TokenSegmentContext() for _ in all_tokens]

    effective_mode = "block" if (has_s and mode == "native") else mode
    effective = SegmentationPolicy(
        mode=effective_mode,
        heuristic=policy.heuristic,
        model=policy.model,
    )
    new_sentences = build_sentences_from_tokens(
        all_tokens,
        contexts,
        effective,
        language=language,
        verbose=verbose,
    )
    _renumber_tokens(new_sentences)

    new_doc = copy.deepcopy(document)
    new_doc.sentences = new_sentences
    new_doc.meta["_segmentation_policy"] = effective_mode
    return new_doc


def policy_for_backend(
    backend_type: str,
    *,
    flexitag_policy: Optional[SegmentationPolicy] = None,
    conllu_policy: Optional[SegmentationPolicy] = None,
) -> SegmentationPolicy:
    backend_key = (backend_type or "").lower()
    if backend_key == "flexitag":
        return flexitag_policy or default_policy_for_target("flexitag")
    return conllu_policy or default_policy_for_target("conllu")
