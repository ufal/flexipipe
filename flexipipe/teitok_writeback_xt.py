"""TEITOK I/O via xmltokenizer: reader (extract + nlp plaintext) and writer (fold).

Architecture (avoids plaintext coupling between flexipipe and xmltokenizer):

1. **xmltokenizer** — ``extract`` + ``build_nlp_plaintext`` → canonical NLP input + xml standoff
2. **flexipipe** — run backends on that ``nlp_plaintext`` → ``Document`` / CoNLL-U standoff
3. **xmltokenizer** — ``attach_conllu`` + ``fold`` → tokenized XML

Flexipipe does not use ``extract_plaintext_for_teitok_backend`` on this path.

Install: ``pip install -e ".[xmltokenizer]"`` from flexipipe (uses sibling ``../xmltokenizer``
when present) or ``pip install -e /path/to/xmltokenizer`` first.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    from lxml import etree as ET
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    HAS_LXML = False

from .conllu import document_to_conllu, file_level_attr_dict
from .doc import Document
from .insert_tokens import verify_structure_preserved


class XmltokenizerWritebackError(Exception):
    """xmltokenizer writeback failed."""


class XmltokenizerNotAvailable(XmltokenizerWritebackError):
    """The xmltokenizer package is not installed."""


# When ``align_debug`` is on (``--debug``), malformed folded XML is written here
# before raising, so the failure line/column can be inspected off-line.
DEBUG_FOLD_DUMP_PATH = "/tmp/wrong.xml"


def _debug_dump_enabled(document: Document, align_debug: bool) -> bool:
    """True when ``--debug`` (or explicit ``_flexipipe_debug`` on document.meta)."""
    if align_debug:
        return True
    return bool(document.meta.get("_flexipipe_debug"))


def _dump_folded_xml_for_debug(
    folded_bytes: bytes,
    document: Document,
    *,
    align_debug: bool,
    dump_path: Optional[str] = None,
) -> Optional[Path]:
    if not _debug_dump_enabled(document, align_debug):
        return None
    path = Path(dump_path or DEBUG_FOLD_DUMP_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(folded_bytes)
    except OSError as err:
        print(
            f"[flexipipe] DEBUG: could not write malformed folded XML to {path}: {err}",
            file=sys.stderr,
        )
        return None
    print(
        f"[flexipipe] DEBUG: wrote malformed folded XML to {path} ({len(folded_bytes)} bytes)",
        file=sys.stderr,
    )
    document.meta["_xt_fold_debug_dump"] = str(path)
    return path


def _parse_folded_xml(
    folded_bytes: bytes, document: Document, *, align_debug: bool
) -> ET.Element:
    try:
        if HAS_LXML:
            parser = ET.XMLParser(strip_cdata=False, remove_blank_text=False)
            return ET.fromstring(folded_bytes, parser)
        return ET.fromstring(folded_bytes)
    except Exception as exc:
        dumped = _dump_folded_xml_for_debug(
            folded_bytes, document, align_debug=align_debug
        )
        msg = f"not well-formed ({exc})"
        if dumped is not None:
            msg = f"{msg}; debug dump → {dumped}"
        elif _debug_dump_enabled(document, align_debug):
            msg = (
                f"{msg}; debug dump to {DEBUG_FOLD_DUMP_PATH} was requested "
                "but the write failed (see stderr)"
            )
        raise XmltokenizerWritebackError(msg) from exc


@dataclass
class XmltokenizerSession:
    """Cached extract state for one TEITOK file (same process, writeback pass)."""

    metadata: Any
    profile: dict
    profile_name: str
    ns_transform: Any
    source_path: str
    nlp_plaintext: str
    # (start, end) offsets in ``nlp_plaintext`` per ``metadata.scope_roots`` entry
    scope_nlp_spans: list[tuple[int, int]]
    # Deactivated bytes passed to ``extract`` (layout-normalized when enabled).
    input_bytes_de: bytes
    layout_normalized: bool = False


def xmltokenizer_available() -> bool:
    try:
        import xmltokenizer  # noqa: F401
        return True
    except ImportError:
        return False


def teitok_layout_options_from_args(args: Any) -> dict[str, Any]:
    """CLI ``--option tei-layout:…`` → kwargs for ``open_xmltokenizer_session``."""
    from .cli_options import parse_flexipipe_options, tei_layout_from_parsed_options

    parsed = getattr(args, "_parsed_options", None)
    if parsed is None:
        parsed = parse_flexipipe_options(list(getattr(args, "option", None) or []))
    return tei_layout_from_parsed_options(parsed)


def apply_teitok_layout_meta(document: Document, args: Any) -> None:
    """Store layout-normalize settings on ``document.meta`` for writeback."""
    opts = teitok_layout_options_from_args(args)
    document.meta["_teitok_layout_normalized"] = opts["layout_normalize"]
    document.meta["_teitok_layout_separator"] = opts["layout_separator"]
    if opts["layout_block_tags"]:
        document.meta["_teitok_layout_block_tags"] = opts["layout_block_tags"]


def resolve_writeback_engine(requested: str) -> str:
    """Return ``xmltokenizer`` or ``flexipipe``."""
    mode = (requested or "auto").strip().lower()
    if mode == "auto":
        return "xmltokenizer" if xmltokenizer_available() else "flexipipe"
    if mode in ("xmltokenizer", "xt"):
        if not xmltokenizer_available():
            raise XmltokenizerNotAvailable(
                "writeback-engine xmltokenizer requested but xmltokenizer is not installed "
                '(pip install -e ".[xmltokenizer]" from flexipipe, or pip install -e /path/to/xmltokenizer)'
            )
        return "xmltokenizer"
    if mode in ("flexipipe", "standoff", "native"):
        return "flexipipe"
    raise ValueError(
        f"unknown writeback-engine {requested!r}; use auto, flexipipe, or xmltokenizer"
    )


def open_xmltokenizer_session(
    path: str,
    *,
    profile_name: str = "tei",
    layout_normalize: bool = False,
    layout_block_tags: Optional[tuple[str, ...]] = None,
    layout_separator: str = "\n",
) -> XmltokenizerSession:
    """Phase A + A.5: xml standoff and ``nlp_plaintext`` for flexipipe NLP."""
    import xmltokenizer as xt

    from .teitok_layout import layout_normalize_deactivated_bytes

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"TEITOK file not found: {path}")

    raw = path_obj.read_bytes()
    profile = xt.load_profile(profile_name)
    raw_de, ns_transform = xt.deactivate(raw)
    layout_applied = False
    if layout_normalize:
        block_tags = frozenset(layout_block_tags) if layout_block_tags else None
        raw_de = layout_normalize_deactivated_bytes(
            raw_de, block_tags=block_tags, separator=layout_separator
        )
        layout_applied = True
    metadata = xt.extract(raw_de, profile, source_path=str(path_obj))

    if not metadata.scope_roots:
        raise XmltokenizerWritebackError("xmltokenizer extract found no scope roots")

    nlp_parts: list[str] = []
    scope_spans: list[tuple[int, int]] = []
    offset = 0
    for root in metadata.scope_roots:
        xt.build_nlp_plaintext(root, profile)
        text = root.nlp_plaintext
        if not text.strip():
            scope_spans.append((offset, offset))
            continue
        if nlp_parts:
            offset += 2  # ``\n\n`` between scopes
        start = offset
        end = offset + len(text)
        scope_spans.append((start, end))
        nlp_parts.append(text)
        offset = end

    if not nlp_parts:
        raise XmltokenizerWritebackError("xmltokenizer extract produced empty nlp_plaintext")

    nlp_joined = "\n\n".join(nlp_parts)

    return XmltokenizerSession(
        metadata=metadata,
        profile=profile,
        profile_name=profile_name,
        ns_transform=ns_transform,
        source_path=str(path_obj),
        nlp_plaintext=nlp_joined,
        scope_nlp_spans=scope_spans,
        input_bytes_de=raw_de,
        layout_normalized=layout_applied,
    )


def nlp_plaintext_for_flexipipe(
    path: str,
    *,
    writeback_engine: str = "auto",
    profile_name: str = "tei",
    textnode_xpath: str = ".//text",
    include_notes: bool = False,
    rejoin_linebreaks: bool = True,
    unicode_normalize: Optional[str] = None,
    layout_normalize: bool = False,
    layout_block_tags: Optional[tuple[str, ...]] = None,
    layout_separator: str = "\n",
) -> Tuple[str, Optional[XmltokenizerSession]]:
    """Plaintext for flexipipe backends: xt ``nlp_plaintext`` or flexipipe extractor."""
    if resolve_writeback_engine(writeback_engine) == "xmltokenizer":
        session = open_xmltokenizer_session(
            path,
            profile_name=profile_name,
            layout_normalize=layout_normalize,
            layout_block_tags=layout_block_tags,
            layout_separator=layout_separator,
        )
        return session.nlp_plaintext, session
    from .insert_tokens import extract_plaintext_for_teitok_backend

    return (
        extract_plaintext_for_teitok_backend(
            path,
            textnode_xpath=textnode_xpath,
            include_notes=include_notes,
            rejoin_linebreaks=rejoin_linebreaks,
            unicode_normalize=unicode_normalize,
        ),
        None,
    )


def _build_change_metadata(document: Document) -> Tuple[str, str]:
    backends_used = document.meta.get("_backends_used", []) or ["flexipipe"]
    if not isinstance(backends_used, list):
        backends_used = ["flexipipe"]
    backends_used = [str(b) for b in backends_used if b is not None]
    file_level_attrs = file_level_attr_dict(document)
    model_keys = sorted(k for k in file_level_attrs if k.endswith("_model"))
    model_str = file_level_attrs[model_keys[0]] if model_keys else None
    backend_names = [b.upper() for b in backends_used]
    change_source = (
        ", ".join(backend_names)
        if len(backend_names) > 1
        else (model_str or (backend_names[0] if backend_names else "flexipipe"))
    )
    tasks: set[str] = set()
    if document.meta.get("_tokenized", False):
        tasks.add("tokenize")
    if document.meta.get("_segmented", False):
        tasks.add("segment")
    if any(t.lemma for s in document.sentences for t in s.tokens):
        tasks.add("lemmatize")
    if any(t.xpos or t.upos for s in document.sentences for t in s.tokens):
        tasks.add("tag")
    if any(t.head for s in document.sentences for t in s.tokens):
        tasks.add("parse")
    if getattr(document, "spans", None) and document.spans.get("ner"):
        tasks.add("ner")
    elif any(getattr(s, "entities", None) for s in document.sentences):
        tasks.add("ner")
    if any(
        t.reg or t.expan or t.mod or t.trslit or t.ltrslit or t.corr or t.lex
        or (t.misc and t.misc != "_")
        for s in document.sentences
        for t in s.tokens
    ):
        tasks.add("normalize")
    tasks_summary_str = ",".join(sorted(tasks)) if tasks else "segment,tokenize"
    change_text = f"Tagged via {change_source} (tasks={tasks_summary_str})"
    return change_text, tasks_summary_str


def _postprocess_output_tree(root: ET.Element, document: Document) -> None:
    from datetime import datetime

    from .teitok import _add_change_to_tei_header
    from .teitok_name_wrap import apply_name_wrappers_to_tree

    raw_fl = document.meta.get("_file_level_attrs")
    if raw_fl is not None and not isinstance(raw_fl, dict):
        print(
            "[flexipipe] WARNING: document.meta['_file_level_attrs'] is not a dict "
            f"({type(raw_fl).__name__}); skipping file-level model attrs in TEI header",
            file=sys.stderr,
        )
    change_text, tasks_summary_str = _build_change_metadata(document)
    change_when = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    apply_name_wrappers_to_tree(root, document)
    _apply_teitok_attribute_mapping(root, document)
    _add_change_to_tei_header(root, change_text, change_when, tasks=tasks_summary_str)


def _preferred_attr_name(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    if "," in value:
        first = value.split(",", 1)[0].strip()
        return first or fallback
    return value.strip() or fallback


def _local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _token_values(token) -> dict[str, str]:
    attrs = token.attrs if isinstance(token.attrs, dict) else {}
    return {
        "xpos": token.xpos or attrs.get("xpos", ""),
        "lemma": token.lemma or attrs.get("lemma", ""),
        "reg": token.reg or attrs.get("reg", ""),
        "expan": token.expan or attrs.get("expan", ""),
        "trslit": token.trslit or attrs.get("trslit", ""),
    }


def _apply_teitok_attribute_mapping(root: ET.Element, document: Document) -> None:
    """Map writeback attrs to TEITOK names (e.g. xpos->pos) and copy extra attrs."""
    attr_map = document.meta.get("_teitok_attr_map")
    if not isinstance(attr_map, dict):
        attr_map = {}

    target = {
        "xpos": _preferred_attr_name(attr_map.get("xpos"), "xpos"),
        "lemma": _preferred_attr_name(attr_map.get("lemma"), "lemma"),
        "reg": _preferred_attr_name(attr_map.get("reg"), "reg"),
        "expan": _preferred_attr_name(attr_map.get("expan"), "expan"),
        "trslit": _preferred_attr_name(attr_map.get("trslit"), "trslit"),
    }

    tok_elems = [e for e in root.iter() if _local_tag(e) == "tok"]
    flat_tokens = _flatten_surface_tokens(document)
    n = min(len(tok_elems), len(flat_tokens))
    for i in range(n):
        elem = tok_elems[i]
        vals = _token_values(flat_tokens[i])
        for key, out_name in target.items():
            value = vals.get(key, "")
            if value and value != "_":
                elem.set(out_name, value)
            elif out_name in elem.attrib and key in {"reg", "expan", "trslit"}:
                elem.attrib.pop(out_name, None)
    # If mapped XPOS name differs from "xpos", remove the internal alias everywhere.
    if target["xpos"] != "xpos":
        for elem in tok_elems:
            elem.attrib.pop("xpos", None)

    has_sentence_tags = any(_local_tag(e) == "s" for e in root.iter())
    if not has_sentence_tags:
        for elem in tok_elems:
            elem.attrib.pop("ord", None)


def _flatten_surface_tokens(document: Document) -> list:
    """Return parent tokens in document order (omit MWT subtokens)."""
    from .doc import Token

    sub_ids: set[int] = set()
    for sent in document.sentences:
        for tok in sent.tokens:
            if tok.is_mwt and tok.subtokens:
                for st in tok.subtokens[1:]:
                    if st.id:
                        sub_ids.add(st.id)
    flat: list[Token] = []
    for sent in document.sentences:
        for tok in sent.tokens:
            if tok.id and tok.id in sub_ids:
                continue
            flat.append(tok)
    return flat


def sync_document_spacing_to_nlp_plaintext(document: Document, nlp_plaintext: str) -> None:
    """Set ``space_after`` on tokens so CoNLL-U aligns with xmltokenizer ``nlp_plaintext``."""
    import xmltokenizer as xt

    from .conllu import document_to_conllu

    flat = _flatten_surface_tokens(document)
    if not flat:
        return

    aligned = xt.align_to_plaintext(
        xt.parse_conllu(
            document_to_conllu(document, model_info=None, create_implicit_mwt=False)
        ),
        nlp_plaintext,
    )
    surface_aligned = [at for at in aligned if at.nlp_start < at.nlp_end]
    if len(flat) != len(surface_aligned):
        return

    for i, (tok, at) in enumerate(zip(flat, surface_aligned)):
        if i + 1 < len(surface_aligned):
            gap = nlp_plaintext[at.nlp_end : surface_aligned[i + 1].nlp_start]
            tok.space_after = bool(gap) and gap.isspace()
        else:
            tok.space_after = None


def document_from_xmltokenizer_nlp_plaintext(
    nlp_plaintext: str,
    *,
    doc_id: str = "",
) -> Document:
    """
    Build a flexipipe ``Document`` whose tokenization matches xmltokenizer fold.

    Uses the xmltokenizer naive tokenizer (same punctuation/paragraph rules as
    ``attach_conllu`` alignment) so flexipipe NLP output can fold back into XML.
    """
    from xmltokenizer.backends.naive import NaiveBackend

    from .conllu import conllu_to_document

    conllu = NaiveBackend().tokenize(nlp_plaintext)
    document = conllu_to_document(conllu, doc_id=doc_id or "")
    sync_document_spacing_to_nlp_plaintext(document, nlp_plaintext)
    document.meta["_tokenized"] = True
    return document


def _conllu_for_scope_nlp(
    document: Document,
    scope_start: int,
    scope_end: int,
    full_nlp: str,
    *,
    create_implicit_mwt: bool,
) -> str:
    """Build CoNLL-U for tokens whose spans fall in ``[scope_start, scope_end)``."""
    if scope_start == 0 and scope_end >= len(full_nlp):
        return document_to_conllu(
            document, model_info=None, create_implicit_mwt=create_implicit_mwt
        )

    import xmltokenizer as xt

    sentences = xt.parse_conllu(
        document_to_conllu(document, model_info=None, create_implicit_mwt=create_implicit_mwt)
    )
    aligned = xt.align_to_plaintext(sentences, full_nlp)
    in_scope = [
        at
        for at in aligned
        if at.nlp_start < scope_end
        and at.nlp_end > scope_start
        and at.nlp_start < at.nlp_end
    ]
    if not in_scope:
        raise XmltokenizerWritebackError(
            "no CoNLL-U tokens align to this scope's nlp_plaintext slice"
        )

    from .doc import Document as FpDocument, Sentence, Token

    sub = FpDocument(id=document.id)
    by_sent: dict[str, list] = {}
    order: list[str] = []
    for at in in_scope:
        if at.sent_id not in by_sent:
            by_sent[at.sent_id] = []
            order.append(at.sent_id)
        by_sent[at.sent_id].append(at)

    for sid in order:
        sent = Sentence(sent_id=sid)
        for at in by_sent[sid]:
            ctok = at.ctok
            head_raw = ctok.head
            try:
                head_int = int(head_raw) if head_raw and head_raw != "_" else 0
            except ValueError:
                head_int = 0
            sent.tokens.append(
                Token(
                    id=len(sent.tokens) + 1,
                    form=ctok.form,
                    lemma=ctok.lemma if ctok.lemma != "_" else "",
                    upos=ctok.upos if ctok.upos != "_" else "",
                    xpos=ctok.xpos if ctok.xpos != "_" else "",
                    feats=ctok.feats if ctok.feats != "_" else "",
                    head=head_int,
                    deprel=ctok.deprel if ctok.deprel != "_" else "",
                    misc=ctok.misc if ctok.misc != "_" else "",
                )
            )
        sub.sentences.append(sent)

    return document_to_conllu(sub, model_info=None, create_implicit_mwt=create_implicit_mwt)


def writeback_teitok_with_xmltokenizer(
    document: Document,
    original_path: str,
    output_path: Optional[str] = None,
    *,
    session: Optional[XmltokenizerSession] = None,
    profile_name: str = "tei",
    align_debug: bool = False,
    run_xt_validate: bool = True,
    create_implicit_mwt: bool = True,
) -> None:
    """Fold flexipipe's CoNLL-U standoff back into XML via xmltokenizer."""
    import xmltokenizer as xt

    original_path_obj = Path(original_path)
    output_path_obj = Path(output_path) if output_path else original_path_obj

    if session is None:
        session = open_xmltokenizer_session(
            original_path,
            profile_name=profile_name,
            layout_normalize=document.meta.get("_teitok_layout_normalized", False),
            layout_block_tags=document.meta.get("_teitok_layout_block_tags"),
            layout_separator=document.meta.get("_teitok_layout_separator", "\n"),
        )
    elif session.source_path != str(original_path_obj.resolve()):
        session = open_xmltokenizer_session(
            original_path,
            profile_name=profile_name,
            layout_normalize=document.meta.get(
                "_teitok_layout_normalized", session.layout_normalized
            ),
            layout_block_tags=document.meta.get("_teitok_layout_block_tags"),
            layout_separator=document.meta.get("_teitok_layout_separator", "\n"),
        )

    import xmltokenizer as xt

    raw_de_snap = session.input_bytes_de
    if HAS_LXML:
        parser = ET.XMLParser(strip_cdata=False, remove_blank_text=False)
        original_root_snapshot = copy.deepcopy(
            ET.fromstring(raw_de_snap, parser)
        )
    else:
        original_root_snapshot = copy.deepcopy(ET.fromstring(raw_de_snap))

    full_nlp = session.nlp_plaintext
    if document.meta.get("_teitok_extracted_nlp") != full_nlp:
        print(
            "[flexipipe] xmltokenizer writeback: document NLP plaintext differs from "
            "session (NLP may have run on a different extractor)",
            file=sys.stderr,
        )

    w_counter = [0]
    s_counter = [0]
    metadata = session.metadata
    profile = session.profile

    for i, root in enumerate(metadata.scope_roots):
        if not root.nlp_plaintext.strip():
            continue
        span = session.scope_nlp_spans[i] if i < len(session.scope_nlp_spans) else (0, len(full_nlp))
        scope_conllu = _conllu_for_scope_nlp(
            document,
            span[0],
            span[1],
            full_nlp,
            create_implicit_mwt=create_implicit_mwt,
        )
        try:
            xt.attach_conllu(
                root,
                scope_conllu,
                profile=profile,
                w_counter=w_counter,
                s_counter=s_counter,
            )
        except xt.AlignmentError as exc:
            raise XmltokenizerWritebackError(
                f"CoNLL-U alignment failed for scope: {exc}"
            ) from exc
        except xt.CoNLLUParseError as exc:
            raise XmltokenizerWritebackError(f"CoNLL-U parse failed: {exc}") from exc

    out_de = xt.fold(metadata)

    if run_xt_validate:
        try:
            xt.validate(session.input_bytes_de, out_de, metadata)
        except xt.ValidationError as exc:
            raise XmltokenizerWritebackError(f"xmltokenizer validate failed: {exc}") from exc

    # Parse/postprocess in deactivated form (matches extract snapshot for structure checks).
    new_root = _parse_folded_xml(out_de, document, align_debug=align_debug)

    verify_structure_preserved(
        original_root_snapshot,
        new_root,
        ignore_tags={"s", "tok", "dtok"},
        ignore_attrs={"id", "rpt", "cont"},
        debug=align_debug,
    )
    _postprocess_output_tree(new_root, document)

    if HAS_LXML:
        folded_after = ET.tostring(
            new_root,
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=False,
        )
    else:
        folded_after = ET.tostring(new_root, encoding="utf-8", xml_declaration=True)

    # Restore xmlns on disk when we deactivated on input (TEITOK corpora).
    # Previously we reactivated then discarded the result and wrote xmlnsoff= XML.
    if session.ns_transform is not None:
        out_bytes = xt.reactivate(folded_after, session.ns_transform)
    else:
        out_bytes = folded_after

    output_path_obj.write_bytes(out_bytes)

    import re as _re

    tok_count = len(_re.findall(br"<tok[\s/>]", out_bytes))
    if tok_count == 0:
        print(
            f"[flexipipe] WARNING: xmltokenizer writeback wrote {output_path_obj} "
            "with no <tok> elements — check tokenization/alignment",
            file=sys.stderr,
        )
    elif align_debug:
        print(
            f"[flexipipe] xmltokenizer writeback OK → {output_path_obj} "
            f"({tok_count} <tok>)",
            file=sys.stderr,
        )
