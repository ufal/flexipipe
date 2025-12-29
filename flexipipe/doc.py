from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# Standard UD attributes as defined by Universal Dependencies
UD_DOCUMENT_ATTRIBUTES = {
    "newdoc_id": "Unique document identifier",
    "title": "Document title",
    "author": "Document author",
    "date": "Publication/creation date",
    "genre": "Text genre (news, social, fiction, academic, legal, spoken)",
    "publisher": "Publisher information",
    "url": "Source URL",
    "license": "Licensing information",
    "source": "Original source description",
}

UD_PARAGRAPH_ATTRIBUTES = {
    "newpar_id": "Paragraph identifier",
    "section": "Section heading",
    "align": "Alignment information for parallel texts",
}

UD_SENTENCE_ATTRIBUTES = {
    "sent_id": "Unique sentence identifier",
    "text": "Original sentence text",
    "lang": "Language code (ISO 639-1)",
    "date": "Sentence-specific timestamp",
    "speaker": "For spoken texts",
    "participant": "Conversation participant",
    "annotator": "Annotation information",
    "translation": "Translation",
    "align": "Alignment",
    "translation_lang": "Translation language",
}


def _normalize_attrs(attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not attrs:
        return {}
    return dict(attrs)


class AttrsMixin:
    attrs: Dict[str, Any]

    def get_attr(self, name: str, default: str = "") -> str:
        value = self.attrs.get(name)
        return default if value is None else value

    def set_attr(self, name: str, value: Optional[str]) -> None:
        if value is None or value == "":
            self.attrs.pop(name, None)
        else:
            self.attrs[name] = value

    def clear_attr(self, name: str) -> None:
        self.attrs.pop(name, None)

    @staticmethod
    def _coerce_init_value(value: Optional[Any]) -> Optional[str]:
        if isinstance(value, property):
            return None
        return value

@dataclass
class Span(AttrsMixin):
    label: str
    start: int
    end: int
    attrs: Dict[str, Any] = field(default_factory=dict)
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None

    def __post_init__(self):
        self.attrs = _normalize_attrs(self.attrs)

    @classmethod
    def from_dict(cls, data: dict) -> "Span":
        return cls(
            label=data.get("label", ""),
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            attrs=_normalize_attrs(data.get("attrs")),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            byte_start=data.get("byte_start"),
            byte_end=data.get("byte_end"),
        )

    def to_dict(self) -> dict:
        result = {
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }
        if self.attrs:
            result["attrs"] = dict(self.attrs)
        if self.char_start is not None:
            result["char_start"] = self.char_start
        if self.char_end is not None:
            result["char_end"] = self.char_end
        if self.byte_start is not None:
            result["byte_start"] = self.byte_start
        if self.byte_end is not None:
            result["byte_end"] = self.byte_end
        return result


@dataclass
class SubToken(AttrsMixin):
    id: int
    form: str
    lemma: str = ""
    xpos: str = ""
    upos: str = ""
    feats: str = ""
    source: str = ""
    source_id: str = ""
    space_after: bool = True
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    upos_confidence: Optional[float] = None
    xpos_confidence: Optional[float] = None
    lemma_confidence: Optional[float] = None
    deprel_confidence: Optional[float] = None
    reg: InitVar[Optional[str]] = ""
    expan: InitVar[Optional[str]] = ""
    mod: InitVar[Optional[str]] = ""
    trslit: InitVar[Optional[str]] = ""
    ltrslit: InitVar[Optional[str]] = ""
    corr: InitVar[Optional[str]] = ""
    lex: InitVar[Optional[str]] = ""
    tokid: InitVar[Optional[str]] = ""

    def __post_init__(
        self,
        reg: Optional[str],
        expan: Optional[str],
        mod: Optional[str],
        trslit: Optional[str],
        ltrslit: Optional[str],
        corr: Optional[str],
        lex: Optional[str],
        tokid: Optional[str],
    ):
        self.attrs = _normalize_attrs(self.attrs)
        for name, value in (
            ("reg", reg),
            ("expan", expan),
            ("mod", mod),
            ("trslit", trslit),
            ("ltrslit", ltrslit),
            ("corr", corr),
            ("lex", lex),
        ):
            coerced = self._coerce_init_value(value)
            if coerced:
                self.set_attr(name, coerced)
        inferred_tokid = self._coerce_init_value(tokid) or self.attrs.get("tokid") or self.source_id
        if inferred_tokid:
            self.tokid = inferred_tokid

    @property
    def tokid(self) -> str:
        return self.source_id or self.attrs.get("tokid", "")

    @tokid.setter
    def tokid(self, value: Optional[str]) -> None:
        normalized = value or ""
        self.source_id = normalized
        if normalized:
            self.attrs["tokid"] = normalized
        else:
            self.attrs.pop("tokid", None)

    @property
    def reg(self) -> str:
        return self.get_attr("reg")

    @reg.setter
    def reg(self, value: Optional[str]) -> None:
        self.set_attr("reg", value)

    @property
    def expan(self) -> str:
        return self.get_attr("expan")

    @expan.setter
    def expan(self, value: Optional[str]) -> None:
        self.set_attr("expan", value)

    @property
    def mod(self) -> str:
        return self.get_attr("mod")

    @mod.setter
    def mod(self, value: Optional[str]) -> None:
        self.set_attr("mod", value)

    @property
    def trslit(self) -> str:
        return self.get_attr("trslit")

    @trslit.setter
    def trslit(self, value: Optional[str]) -> None:
        self.set_attr("trslit", value)

    @property
    def ltrslit(self) -> str:
        return self.get_attr("ltrslit")

    @ltrslit.setter
    def ltrslit(self, value: Optional[str]) -> None:
        self.set_attr("ltrslit", value)

    @property
    def corr(self) -> str:
        return self.get_attr("corr")

    @corr.setter
    def corr(self, value: Optional[str]) -> None:
        self.set_attr("corr", value)

    @property
    def lex(self) -> str:
        return self.get_attr("lex")

    @lex.setter
    def lex(self, value: Optional[str]) -> None:
        self.set_attr("lex", value)

    @classmethod
    def from_dict(cls, data: dict) -> "SubToken":
        attrs = _normalize_attrs(data.get("attrs"))
        return cls(
            id=int(data.get("id", 0)),
            form=data.get("form", ""),
            lemma=data.get("lemma", ""),
            xpos=data.get("xpos", ""),
            upos=data.get("upos", ""),
            feats=data.get("feats", ""),
            source=data.get("source", ""),
            source_id=data.get("source_id", data.get("tokid", "")),
            space_after=bool(data.get("space_after", True)),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            byte_start=data.get("byte_start"),
            byte_end=data.get("byte_end"),
            attrs=attrs,
            upos_confidence=data.get("upos_confidence"),
            xpos_confidence=data.get("xpos_confidence"),
            lemma_confidence=data.get("lemma_confidence"),
            deprel_confidence=data.get("deprel_confidence"),
            reg=data.get("reg"),
            expan=data.get("expan"),
            mod=data.get("mod"),
            trslit=data.get("trslit"),
            ltrslit=data.get("ltrslit"),
            corr=data.get("corr"),
            lex=data.get("lex"),
            tokid=data.get("tokid") or attrs.get("tokid"),
        )

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "form": self.form,
            "lemma": self.lemma,
            "xpos": self.xpos,
            "upos": self.upos,
            "feats": self.feats,
            "source": self.source,
            "source_id": self.source_id,
            "space_after": self.space_after,
            "attrs": dict(self.attrs),
        }
        if self.char_start is not None:
            result["char_start"] = self.char_start
        if self.char_end is not None:
            result["char_end"] = self.char_end
        if self.byte_start is not None:
            result["byte_start"] = self.byte_start
        if self.byte_end is not None:
            result["byte_end"] = self.byte_end
        for key in ("reg", "expan", "mod", "trslit", "ltrslit", "corr", "lex"):
            value = self.get_attr(key)
            if value:
                result[key] = value
        tokid = self.tokid
        if tokid:
            result["tokid"] = tokid
        if self.upos_confidence is not None:
            result["upos_confidence"] = self.upos_confidence
        if self.xpos_confidence is not None:
            result["xpos_confidence"] = self.xpos_confidence
        if self.lemma_confidence is not None:
            result["lemma_confidence"] = self.lemma_confidence
        if self.deprel_confidence is not None:
            result["deprel_confidence"] = self.deprel_confidence
        return result


@dataclass
class Token(AttrsMixin):
    id: int
    form: str
    lemma: str = ""
    xpos: str = ""
    upos: str = ""
    feats: str = ""
    is_mwt: bool = False
    mwt_start: int = 0
    mwt_end: int = 0
    parts: List[str] = field(default_factory=list)
    subtokens: List[SubToken] = field(default_factory=list)
    source: str = ""
    source_id: str = ""
    head: int = 0
    deprel: str = ""
    deps: str = ""
    misc: str = ""
    space_after: Optional[bool] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    upos_confidence: Optional[float] = None
    xpos_confidence: Optional[float] = None
    lemma_confidence: Optional[float] = None
    deprel_confidence: Optional[float] = None
    reg: InitVar[Optional[str]] = ""
    expan: InitVar[Optional[str]] = ""
    mod: InitVar[Optional[str]] = ""
    trslit: InitVar[Optional[str]] = ""
    ltrslit: InitVar[Optional[str]] = ""
    corr: InitVar[Optional[str]] = ""
    lex: InitVar[Optional[str]] = ""
    tokid: InitVar[Optional[str]] = ""

    def __post_init__(
        self,
        reg: Optional[str],
        expan: Optional[str],
        mod: Optional[str],
        trslit: Optional[str],
        ltrslit: Optional[str],
        corr: Optional[str],
        lex: Optional[str],
        tokid: Optional[str],
    ):
        self.attrs = _normalize_attrs(self.attrs)
        for name, value in (
            ("reg", reg),
            ("expan", expan),
            ("mod", mod),
            ("trslit", trslit),
            ("ltrslit", ltrslit),
            ("corr", corr),
            ("lex", lex),
        ):
            coerced = self._coerce_init_value(value)
            if coerced:
                self.set_attr(name, coerced)
        inferred_tokid = self._coerce_init_value(tokid) or self.attrs.get("tokid") or self.source_id
        if inferred_tokid:
            self.tokid = inferred_tokid

    @property
    def tokid(self) -> str:
        return self.source_id or self.attrs.get("tokid", "")

    @tokid.setter
    def tokid(self, value: Optional[str]) -> None:
        normalized = value or ""
        self.source_id = normalized
        if normalized:
            self.attrs["tokid"] = normalized
        else:
            self.attrs.pop("tokid", None)

    @property
    def reg(self) -> str:
        return self.get_attr("reg")

    @reg.setter
    def reg(self, value: Optional[str]) -> None:
        self.set_attr("reg", value)

    @property
    def expan(self) -> str:
        return self.get_attr("expan")

    @expan.setter
    def expan(self, value: Optional[str]) -> None:
        self.set_attr("expan", value)

    @property
    def mod(self) -> str:
        return self.get_attr("mod")

    @mod.setter
    def mod(self, value: Optional[str]) -> None:
        self.set_attr("mod", value)

    @property
    def trslit(self) -> str:
        return self.get_attr("trslit")

    @trslit.setter
    def trslit(self, value: Optional[str]) -> None:
        self.set_attr("trslit", value)

    @property
    def ltrslit(self) -> str:
        return self.get_attr("ltrslit")

    @ltrslit.setter
    def ltrslit(self, value: Optional[str]) -> None:
        self.set_attr("ltrslit", value)

    @property
    def corr(self) -> str:
        return self.get_attr("corr")

    @corr.setter
    def corr(self, value: Optional[str]) -> None:
        self.set_attr("corr", value)

    @property
    def lex(self) -> str:
        return self.get_attr("lex")

    @lex.setter
    def lex(self, value: Optional[str]) -> None:
        self.set_attr("lex", value)

    @classmethod
    def from_dict(cls, data: dict) -> "Token":
        subtokens = [SubToken.from_dict(d) for d in data.get("subtokens", [])]
        parts = list(data.get("parts", []))
        attrs = _normalize_attrs(data.get("attrs"))
        return cls(
            id=int(data.get("id", 0)),
            form=data.get("form", ""),
            lemma=data.get("lemma", ""),
            xpos=data.get("xpos", ""),
            upos=data.get("upos", ""),
            feats=data.get("feats", ""),
            is_mwt=bool(data.get("is_mwt", False)),
            mwt_start=int(data.get("mwt_start", 0)),
            mwt_end=int(data.get("mwt_end", 0)),
            parts=parts,
            subtokens=subtokens,
            source=data.get("source", ""),
            source_id=data.get("source_id", data.get("tokid", "")),
            head=int(data.get("head", 0)),
            deprel=data.get("deprel", ""),
            deps=data.get("deps", ""),
            misc=data.get("misc", ""),
            space_after=data.get("space_after"),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            byte_start=data.get("byte_start"),
            byte_end=data.get("byte_end"),
            attrs=attrs,
            upos_confidence=data.get("upos_confidence"),
            xpos_confidence=data.get("xpos_confidence"),
            lemma_confidence=data.get("lemma_confidence"),
            deprel_confidence=data.get("deprel_confidence"),
            reg=data.get("reg"),
            expan=data.get("expan"),
            mod=data.get("mod"),
            trslit=data.get("trslit"),
            ltrslit=data.get("ltrslit"),
            corr=data.get("corr"),
            lex=data.get("lex"),
            tokid=data.get("tokid") or attrs.get("tokid"),
        )

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "form": self.form,
            "lemma": self.lemma,
            "xpos": self.xpos,
            "upos": self.upos,
            "feats": self.feats,
            "is_mwt": self.is_mwt,
            "mwt_start": self.mwt_start,
            "mwt_end": self.mwt_end,
            "parts": list(self.parts),
            "subtokens": [st.to_dict() for st in self.subtokens],
            "source": self.source,
            "source_id": self.source_id,
            "head": self.head,
            "deprel": self.deprel,
            "deps": self.deps,
            "misc": self.misc,
            "space_after": self.space_after,
            "attrs": dict(self.attrs),
        }
        if self.char_start is not None:
            result["char_start"] = self.char_start
        if self.char_end is not None:
            result["char_end"] = self.char_end
        if self.byte_start is not None:
            result["byte_start"] = self.byte_start
        if self.byte_end is not None:
            result["byte_end"] = self.byte_end
        for key in ("reg", "expan", "mod", "trslit", "ltrslit", "corr", "lex"):
            value = self.get_attr(key)
            if value:
                result[key] = value
        tokid = self.tokid
        if tokid:
            result["tokid"] = tokid
        if self.upos_confidence is not None:
            result["upos_confidence"] = self.upos_confidence
        if self.xpos_confidence is not None:
            result["xpos_confidence"] = self.xpos_confidence
        if self.lemma_confidence is not None:
            result["lemma_confidence"] = self.lemma_confidence
        if self.deprel_confidence is not None:
            result["deprel_confidence"] = self.deprel_confidence
        return result


@dataclass
class Entity:
    """Named entity with span information."""
    start: int  # Token index (1-based) of first token in entity
    end: int    # Token index (1-based) of last token in entity (inclusive)
    label: str  # Entity type (e.g., "PERSON", "ORG", "GPE")
    text: str = ""  # Optional: text of the entity
    attrs: Dict[str, str] = field(default_factory=dict)  # Additional attributes
    
    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            label=data.get("label", ""),
            text=data.get("text", ""),
            attrs=dict(data.get("attrs", {})),
        )
    
    def to_dict(self) -> dict:
        result = {
            "start": self.start,
            "end": self.end,
            "label": self.label,
        }
        if self.text:
            result["text"] = self.text
        if self.attrs:
            result["attrs"] = dict(self.attrs)
        return result


@dataclass
class Sentence(AttrsMixin):
    id: str
    sent_id: str = ""
    text: str = ""
    tokens: List[Token] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    spans: Dict[str, List[Span]] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)
    source_id: str = ""
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    corr: InitVar[Optional[str]] = ""

    def __post_init__(self, corr: Optional[str]):
        self.attrs = _normalize_attrs(self.attrs)
        self.spans = {key: [span if isinstance(span, Span) else Span.from_dict(span) for span in value] for key, value in self.spans.items()}
        coerced = self._coerce_init_value(corr)
        if coerced:
            self.set_attr("corr", coerced)
        # Extract standard UD attributes from attrs if present
        self._extract_standard_attrs()

    def _extract_standard_attrs(self) -> None:
        """Extract standard UD attributes from attrs and keep them accessible."""
        # Standard attributes are kept in attrs but can be accessed via properties
        pass

    def get_standard_attrs(self) -> Dict[str, str]:
        """Get all standard UD sentence attributes that are present."""
        result = {}
        for attr_name in UD_SENTENCE_ATTRIBUTES.keys():
            # sent_id and text are direct fields, not in attrs
            if attr_name == "sent_id":
                if self.sent_id:
                    result[attr_name] = self.sent_id
            elif attr_name == "text":
                if self.text:
                    result[attr_name] = self.text
            else:
                value = self.attrs.get(attr_name)
                if value:
                    result[attr_name] = str(value)
        return result

    def set_standard_attr(self, name: str, value: Optional[str]) -> None:
        """Set a standard UD sentence attribute."""
        if name not in UD_SENTENCE_ATTRIBUTES:
            raise ValueError(f"'{name}' is not a standard UD sentence attribute")
        if name == "sent_id":
            self.sent_id = value or ""
        elif name == "text":
            self.text = value or ""
        else:
            self.set_attr(name, value)

    @property
    def corr(self) -> str:
        return self.get_attr("corr")

    @corr.setter
    def corr(self, value: Optional[str]) -> None:
        self.set_attr("corr", value)

    @classmethod
    def from_dict(cls, data: dict) -> "Sentence":
        return cls(
            id=data.get("id", ""),
            sent_id=data.get("sent_id", ""),
            text=data.get("text", ""),
            tokens=[Token.from_dict(t) for t in data.get("tokens", [])],
            entities=[Entity.from_dict(e) for e in data.get("entities", [])],
            spans={layer: [Span.from_dict(span) for span in entries] for layer, entries in data.get("spans", {}).items()},
            attrs=_normalize_attrs(data.get("attrs")),
            source_id=data.get("source_id", ""),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            byte_start=data.get("byte_start"),
            byte_end=data.get("byte_end"),
            corr=data.get("corr"),
        )

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "sent_id": self.sent_id,
            "text": self.text,
            "tokens": [tok.to_dict() for tok in self.tokens],
            "attrs": dict(self.attrs),
            "source_id": self.source_id,
        }
        if self.char_start is not None:
            result["char_start"] = self.char_start
        if self.char_end is not None:
            result["char_end"] = self.char_end
        if self.byte_start is not None:
            result["byte_start"] = self.byte_start
        if self.byte_end is not None:
            result["byte_end"] = self.byte_end
        if self.entities:
            result["entities"] = [ent.to_dict() for ent in self.entities]
        if self.spans:
            result["spans"] = {
                layer: [span.to_dict() for span in spans] for layer, spans in self.spans.items()
            }
        corr = self.corr
        if corr:
            result["corr"] = corr
        return result


@dataclass
class Document(AttrsMixin):
    id: str = ""
    sentences: List[Sentence] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)
    spans: Dict[str, List[Span]] = field(default_factory=dict)

    def __post_init__(self):
        self.attrs = _normalize_attrs(self.attrs)
        self.spans = {key: list(value) for key, value in self.spans.items()}
        # Extract standard UD attributes from attrs if present
        self._extract_standard_attrs()

    def _extract_standard_attrs(self) -> None:
        """Extract standard UD attributes from attrs and keep them accessible."""
        # Standard attributes are kept in attrs but can be accessed via properties
        # If id is set and newdoc_id is not, use id as newdoc_id
        if self.id and not self.attrs.get("newdoc_id"):
            self.attrs["newdoc_id"] = self.id

    def get_standard_attrs(self) -> Dict[str, str]:
        """Get all standard UD document attributes that are present."""
        result = {}
        for attr_name in UD_DOCUMENT_ATTRIBUTES.keys():
            # id is a direct field, map it to newdoc_id for standard attributes
            if attr_name == "newdoc_id":
                value = self.attrs.get("newdoc_id") or self.id
                if value:
                    result[attr_name] = str(value)
            else:
                value = self.attrs.get(attr_name)
                if value:
                    result[attr_name] = str(value)
        return result

    def set_standard_attr(self, name: str, value: Optional[str]) -> None:
        """Set a standard UD document attribute."""
        if name not in UD_DOCUMENT_ATTRIBUTES:
            raise ValueError(f"'{name}' is not a standard UD document attribute")
        if name == "newdoc_id":
            self.set_attr("newdoc_id", value)
            # Also update id if it's empty or matches the old newdoc_id
            if value and (not self.id or self.id == self.attrs.get("newdoc_id")):
                self.id = value
        else:
            self.set_attr(name, value)

    @classmethod
    def from_plain_text(
        cls, 
        text: str, 
        *, 
        doc_id: str = "", 
        tokenize: bool = True, 
        segment: bool = True,
        keep_contractions: bool = True,
        keep_hyphenated: bool = True,
    ) -> "Document":
        """
        Create a Document from plain text.
        
        Args:
            text: Plain text input
            doc_id: Document ID
            tokenize: If True, use improved Unicode-based tokenization (properly handles
                     combining marks/diacritics). If False, simple whitespace splitting.
            segment: If True, split text into sentences at sentence-ending punctuation.
            keep_contractions: If True, keep contractions like "it's" as one token.
                             If False, split them: ["it", "'", "s"]
            keep_hyphenated: If True, keep hyphenated compounds like "state-of-the-art"
                           as one token. If False, split on hyphens.
        """
        from .unicode_tokenizer import tokenize_unicode, segment_sentences
        
        document = cls(id=doc_id)
        sentence_id = 1
        
        # Process each line
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            
            # Sentence segmentation
            if segment:
                sentence_texts = segment_sentences(stripped, preserve_quotes=True)
            else:
                sentence_texts = [stripped]
            
            # Process each sentence
            for sent_text in sentence_texts:
                sent_text = sent_text.strip()
                if not sent_text:
                    continue
                
                if tokenize:
                    # Use improved Unicode-based tokenization
                    parts = tokenize_unicode(
                        sent_text,
                        keep_contractions=keep_contractions,
                        keep_hyphenated=keep_hyphenated,
                        keep_trailing_apostrophe=keep_contractions,  # Use same setting
                    )
                else:
                    # Simple whitespace splitting, but also separate trailing punctuation
                    # This ensures periods, exclamation marks, etc. are separate tokens
                    import re
                    # Split on whitespace, then separate trailing punctuation from each word
                    words = sent_text.split()
                    parts = []
                    for word in words:
                        # Check if word ends with sentence-ending punctuation
                        match = re.match(r'^(.+?)([.!?]+)$', word)
                        if match:
                            # Word has trailing punctuation - split it
                            parts.append(match.group(1))  # The word part
                            parts.append(match.group(2))  # The punctuation part
                        else:
                            # No trailing punctuation - keep as is
                            parts.append(word)
                
                if not parts:
                    continue
                
                tokens = []
                for idx, part in enumerate(parts):
                    # Determine space_after: True if not the last token and next token doesn't start with punctuation
                    space_after = True
                    if idx + 1 < len(parts):
                        next_part = parts[idx + 1]
                        # If next token is punctuation, no space after
                        # Check if next token starts with a letter/number (not punctuation)
                        import unicodedata
                        if next_part:
                            next_first_char = next_part[0]
                            next_cat = unicodedata.category(next_first_char)
                            if not (next_cat.startswith(("L", "N"))):
                                space_after = False
                    else:
                        # Last token: set to None (no SpaceAfter entry in CoNLL-U)
                        space_after = None
                    
                    tokens.append(
                        Token(
                            id=idx + 1,
                            form=part,
                            lemma="",
                            space_after=space_after,
                        )
                    )
                
                sentence = Sentence(
                    id=f"s{sentence_id}",
                    sent_id="",
                    text=sent_text,
                    tokens=tokens,
                )
                document.sentences.append(sentence)
                sentence_id += 1
        return document

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        spans_data = {}
        for label, entries in data.get("spans", {}).items():
            spans_data[label] = [Span.from_dict(entry) for entry in entries]
        return cls(
            id=data.get("id", ""),
            sentences=[Sentence.from_dict(s) for s in data.get("sentences", [])],
            meta=dict(data.get("meta", {})),
            attrs=_normalize_attrs(data.get("attrs")),
            spans=spans_data,
        )

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "sentences": [sent.to_dict() for sent in self.sentences],
            "meta": dict(self.meta),
            "attrs": dict(self.attrs),
        }
        if self.spans:
            result["spans"] = {
                label: [span.to_dict() for span in spans] for label, spans in self.spans.items()
            }
        return result

    def normalize_unicode(self, form: str = "none") -> None:
        """
        Normalize Unicode in all token forms and lemmas in-place.
        
        Args:
            form: Unicode normalization form ("none", "NFC", "NFD")
        """
        if form == "none":
            return
        from .unicode_utils import normalize_unicode
        for sentence in self.sentences:
            # Preserve original text if there are no tokens (needed for treetagger)
            original_text = sentence.text if not sentence.tokens else None
            
            for token in sentence.tokens:
                token.form = normalize_unicode(token.form, form) or ""
                token.lemma = normalize_unicode(token.lemma, form) or ""
                if token.subtokens:
                    for subtoken in token.subtokens:
                        subtoken.form = normalize_unicode(subtoken.form, form) or ""
                        subtoken.lemma = normalize_unicode(subtoken.lemma, form) or ""
            # Rebuild sentence text after normalization
            # If there are no tokens, preserve the original text
            if sentence.tokens:
                sentence.text = _rebuild_sentence_text(sentence.tokens)
            elif original_text:
                # No tokens but we had original text - preserve it (normalize the text itself)
                sentence.text = normalize_unicode(original_text, form) or original_text

    def tokens(self) -> Iterable[Token]:
        for sentence in self.sentences:
            yield from sentence.tokens

    def add_span(self, layer: str, span: Span) -> None:
        self.spans.setdefault(layer, []).append(span)

    def iter_spans(self, layer: Optional[str] = None) -> Iterable[Span]:
        if layer is not None:
            for span in self.spans.get(layer, []):
                yield span
        else:
            for spans in self.spans.values():
                for span in spans:
                    yield span


def _rebuild_sentence_text(tokens: Iterable[Token]) -> str:
    parts: List[str] = []
    for token in tokens:
        parts.append(token.form)
        if token.space_after:
            parts.append(" ")
    return "".join(parts).strip()


def _normalized_form(value: str) -> str:
    if value and value not in {"_", "--"}:
        return value
    return ""


def _apply_form_strategy_to_token(token: Token | SubToken, strategy: str) -> None:
    if strategy == "reg":
        replacement = _normalized_form(getattr(token, "reg", ""))
        if replacement:
            token.form = replacement
    if isinstance(token, Token) and token.subtokens:
        for subtoken in token.subtokens:
            _apply_form_strategy_to_token(subtoken, strategy)


def apply_nlpform(document: Document, form_strategy: str) -> Document:
    """
    Mutate a document in-place so that Token.form/SubToken.form follow the requested strategy.

    Args:
        document: Document to update
        form_strategy: "form" (default) or "reg"
    """
    strategy = (form_strategy or "form").lower()
    if strategy == "form":
        return document
    if strategy not in {"reg"}:
        return document

    for sentence in document.sentences:
        for token in sentence.tokens:
            _apply_form_strategy_to_token(token, strategy)
        sentence.text = _rebuild_sentence_text(sentence.tokens)
    return document
