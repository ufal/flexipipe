from __future__ import annotations

from typing import Dict, List, Optional

from .doc import Document, Entity, Sentence, SubToken, Token
from .model_storage import get_use_reg_for_nlp


def collect_span_entities_by_sentence(
    document: Document,
    layer: str = "ner",
) -> Dict[int, List[Entity]]:
    span_map: Dict[int, List[Entity]] = {}
    spans = document.spans.get(layer, []) if getattr(document, "spans", None) else []
    if not spans:
        return span_map
    sentence_bounds: List[tuple[int, int]] = []
    offset = 0
    for sentence in document.sentences:
        length = len(sentence.tokens)
        if length <= 0:
            sentence_bounds.append((offset + 1, offset))
            continue
        start = offset + 1
        end = offset + length
        sentence_bounds.append((start, end))
        offset = end
    for span in spans:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        if start is None or end is None:
            continue
        for sent_idx, (sent_start, sent_end) in enumerate(sentence_bounds):
            if sent_start <= start and end <= sent_end and sent_end >= sent_start:
                local_start = start - sent_start + 1
                local_end = end - sent_start + 1
                if local_start < 1 or local_end < local_start:
                    break
                text = ""
                attrs_dict = {}
                attrs = getattr(span, "attrs", None)
                if attrs:
                    for key, value in attrs.items():
                        if value is None:
                            continue
                        if key == "text" and isinstance(value, str):
                            text = value
                        attrs_dict[key] = value if isinstance(value, str) else str(value)
                if text and not attrs_dict.get("text"):
                    attrs_dict["text"] = text
                entity = Entity(
                    start=local_start,
                    end=local_end,
                    label=getattr(span, "label", "") or "",
                    text=text,
                    attrs=attrs_dict,
                )
                span_map.setdefault(sent_idx, []).append(entity)
                break
    return span_map


def _maybe_add_attr(attrs: dict, key: str, value, *, allow_zero: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if not value:
            return
    elif isinstance(value, (int, float)):
        if value == 0 and not allow_zero:
            return
    attrs.setdefault(key, value)


def _subtoken_to_json(subtoken: SubToken) -> dict:
    data = {
        "id": subtoken.id,
        "form": subtoken.form,
        "space_after": subtoken.space_after,
    }
    attrs = dict(subtoken.attrs)
    _maybe_add_attr(attrs, "lemma", subtoken.lemma)
    _maybe_add_attr(attrs, "upos", subtoken.upos)
    _maybe_add_attr(attrs, "xpos", subtoken.xpos)
    _maybe_add_attr(attrs, "feats", subtoken.feats)
    _maybe_add_attr(attrs, "source", subtoken.source)
    _maybe_add_attr(attrs, "tokid", subtoken.tokid)
    _maybe_add_attr(attrs, "upos_confidence", subtoken.upos_confidence, allow_zero=True)
    _maybe_add_attr(attrs, "xpos_confidence", subtoken.xpos_confidence, allow_zero=True)
    _maybe_add_attr(attrs, "lemma_confidence", subtoken.lemma_confidence, allow_zero=True)
    if attrs:
        data["attrs"] = attrs
    return data


def _token_to_json(token: Token) -> dict:
    data = {
        "id": token.id,
        "form": token.form,
        "is_mwt": bool(token.is_mwt),
        "subtokens": [_subtoken_to_json(sub) for sub in token.subtokens],
        "space_after": token.space_after,
    }
    attrs = dict(token.attrs)
    _maybe_add_attr(attrs, "lemma", token.lemma)
    _maybe_add_attr(attrs, "upos", token.upos)
    _maybe_add_attr(attrs, "xpos", token.xpos)
    _maybe_add_attr(attrs, "feats", token.feats)
    _maybe_add_attr(attrs, "source", token.source)
    _maybe_add_attr(attrs, "tokid", token.tokid)
    _maybe_add_attr(attrs, "upos_confidence", token.upos_confidence, allow_zero=True)
    _maybe_add_attr(attrs, "xpos_confidence", token.xpos_confidence, allow_zero=True)
    _maybe_add_attr(attrs, "lemma_confidence", token.lemma_confidence, allow_zero=True)
    _maybe_add_attr(attrs, "deprel_confidence", token.deprel_confidence, allow_zero=True)
    _maybe_add_attr(attrs, "head", token.head, allow_zero=False)
    _maybe_add_attr(attrs, "deprel", token.deprel)
    _maybe_add_attr(attrs, "deps", token.deps)
    _maybe_add_attr(attrs, "misc", token.misc)
    if attrs:
        data["attrs"] = attrs
    return data


def _sentence_to_json(sentence: Sentence) -> dict:
    data = {
        "id": sentence.id,
        "sent_id": sentence.sent_id,
        "text": sentence.text,
        "tokens": [_token_to_json(tok) for tok in sentence.tokens],
    }
    if sentence.entities:
        data["entities"] = [ent.to_dict() for ent in sentence.entities]
    if sentence.spans:
        data["spans"] = {
            layer: [span.to_dict() for span in spans]
            for layer, spans in sentence.spans.items()
        }
    if sentence.attrs:
        data["attrs"] = dict(sentence.attrs)
    if sentence.source_id:
        data["source_id"] = sentence.source_id
    return data


def document_to_json_payload(document: Document) -> dict:
    result = {
        "id": document.id,
        "sentences": [_sentence_to_json(sent) for sent in document.sentences],
        "meta": dict(document.meta),
        "attrs": dict(document.attrs),
    }
    if document.spans:
        result["spans"] = {layer: [span.to_dict() for span in spans] for layer, spans in document.spans.items()}
    return result


def get_effective_form(token: Token | SubToken, *, use_reg_for_nlp: Optional[bool] = None) -> str:
    """
    Get the effective form to use for NLP processing.
    
    If use_reg_for_nlp is True (or configured), returns token.reg if available,
    otherwise returns token.form. This allows backends to use normalized forms
    when available, similar to how flexitag works.
    
    Args:
        token: Token or SubToken to get form from
        use_reg_for_nlp: Whether to use reg when available. If None, reads from config.
    
    Returns:
        The form to use for NLP processing (reg if available and configured, otherwise form)
    """
    if use_reg_for_nlp is None:
        from .model_storage import get_use_reg_for_nlp
        use_reg_for_nlp = get_use_reg_for_nlp()
    
    if use_reg_for_nlp:
        reg = getattr(token, "reg", "") or token.attrs.get("reg", "")
        if reg and reg != "_" and reg != "":
            return reg
    
    return token.form or ""

