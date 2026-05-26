"""
Version 3: Build plain-text with standoff representation first.

This approach:
1. Extracts plain text from XML (similar to xml2standoff)
2. Creates a standoff representation mapping XML elements to character positions
3. Maps tokens/sentences to character positions in the plain text
4. Rebuilds XML by inserting <s> and <tok> elements at the correct positions
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
import copy
import re

try:
    from lxml import etree as ET
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    HAS_LXML = False

# Import Document and Token types
try:
    from .doc import Document, Token, Sentence
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from flexipipe.doc import Document, Token, Sentence

from .linebreak_rejoin import (
    RejoinSpan,
    apply_rejoin_to_token_positions,
    build_nlp_plaintext,
    detect_rejoin_spans,
    merged_form_for_display_span,
    slice_rejoin_spans_for_block,
)
from .teitok_block_align import (
    _sentence_tokens,
    clip_sentences_to_nlp_range,
    find_sentences_for_block_nlp_slice,
    find_tokens_at,
)


def _teitok_local_tag(elem: ET.Element) -> str:
    """Local tag name (no namespace)."""
    t = elem.tag
    if not isinstance(t, str):
        return ""
    return t.split("}")[-1].lower()


def _teitok_parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    """Build parent map for all elements under root."""
    pm: Dict[ET.Element, ET.Element] = {}
    for parent in root.iter():
        for child in parent:
            pm[child] = parent
    return pm


def _teitok_move_tokens_into_s(root: ET.Element, block_elements_set: Set[str]) -> None:
    """
    Move following siblings into each <s> until next <s> or block (per xmltokenize.pl).
    Ensures tokens end up inside <s>...</s> for indexing and sameAs.
    """
    parent_map = _teitok_parent_map(root)
    for s in list(root.iter()):
        if _teitok_local_tag(s) != "s":
            continue
        parent = parent_map.get(s)
        if parent is None:
            continue
        children = list(parent)
        try:
            i = children.index(s)
        except ValueError:
            continue
        to_move: List[ET.Element] = []
        for j in range(i + 1, len(children)):
            c = children[j]
            tag = _teitok_local_tag(c)
            if tag == "s" or tag in block_elements_set:
                break
            to_move.append(c)
        for c in to_move:
            parent.remove(c)
            s.append(c)


def _teitok_set_sameas_on_s(root: ET.Element) -> None:
    """Set sameAs on each <s> to #first_tok_id for CQP/indexing."""
    for s in root.iter():
        if _teitok_local_tag(s) != "s":
            continue
        first_tok = None
        for n in s.iter():
            if _teitok_local_tag(n) == "tok":
                first_tok = n
                break
        if first_tok is not None:
            tok_id = first_tok.get("id") or first_tok.get("{http://www.w3.org/XML/1998/namespace}id")
            if tok_id:
                s.set("sameAs", "#" + tok_id)


def _teitok_set_form_deleted(root: ET.Element) -> None:
    """Set form=\"--\" for every <tok> inside <del> (deleted words; -- = no form, not indexed in CQP)."""
    for elem in root.iter():
        if _teitok_local_tag(elem) != "del":
            continue
        for tok in elem.iter():
            if _teitok_local_tag(tok) == "tok":
                tok.set("form", "--")


def insert_tokens_into_teitok(
    document: Document,
    original_path: str,
    output_path: Optional[str] = None,
    textnode_xpath: str = ".//text",
    include_notes: bool = False,
    block_elements: Optional[List[str]] = None,
    extract_elements: Optional[List[str]] = None,
    settings: Optional[Any] = None,
    use_minidom_rebuild: bool = False,
    use_string_rebuild: bool = False,
    use_teitok_rebuild: bool = False,
    align_debug: bool = False,
) -> None:
    """
    Insert tokens and sentences into a non-tokenized TEITOK XML file.

    1. Extract plain text with standoff markup
    2. Map tokens/sentences to character positions
    3. Rebuild XML with <s> and <tok> elements inserted at correct positions

    Rebuild engine (one of):
    - default: in-place lxml/ET (standoff).
    - use_minidom_rebuild: minidom, text as #text nodes (no .tail).
    - use_string_rebuild: serialize block to string (spacing already correct),
      map plaintext positions to XML positions with a simple scan, insert tags,
      re-parse. No .tail handling and no regex for XML structure.
    - use_teitok_rebuild: Python port of TEITOK's xmltokenize.pl (regex/line-based).
    """
    # Set defaults
    if block_elements is None:
        block_elements = ['div', 'head', 'p', 'u', 'speaker']
    if extract_elements is None:
        extract_elements = ['note', 'desc', 'gap', 'fw', 'rdg']
    
    original_path_obj = Path(original_path)
    if not original_path_obj.exists():
        raise FileNotFoundError(f"Original TEITOK file not found: {original_path}")
    
    output_path_obj = Path(output_path) if output_path else original_path_obj

    # TEITOK-style engine: string/regex tokenizer (port of xmltokenize.pl)
    if use_teitok_rebuild:
        from .insert_tokens_teitok import tokenize_teitok_style
        from .teitok import _add_change_to_tei_header
        from datetime import datetime
        raw_xml = original_path_obj.read_text(encoding="utf-8")
        notok = "|".join(extract_elements) if extract_elements else "note|desc|gap|fw|rdg"
        tokenized = tokenize_teitok_style(
            raw_xml,
            text_tag="text",
            block_elements=block_elements,
            notok_elements=notok,
            split_sentences=True,
            keep_ns=False,
        )
        # Parse tokenized XML and merge backend token attributes (lemma, upos, etc.)
        # Pass bytes so the parser accepts the encoding declaration (Unicode + declaration raises)
        root = ET.fromstring(tokenized.encode("utf-8"))
        block_elements_set = {e.lower() for e in (block_elements or [])}
        # DOM move: put following siblings into each <s> until next <s> or block (per xmltokenize.pl)
        _teitok_move_tokens_into_s(root, block_elements_set)
        # sameAs on <s>: point to first token id for CQP/indexing
        _teitok_set_sameas_on_s(root)
        # All <tok> in document order (handle optional namespace)
        def _is_tok(elem: ET.Element) -> bool:
            t = elem.tag
            if t == "tok":
                return True
            return isinstance(t, str) and t.endswith("}tok")
        tok_elems = [e for e in root.iter() if _is_tok(e)]
        backend_tokens = [t for s in document.sentences for t in s.tokens]
        if tok_elems and backend_tokens:
            def _resolve_attr(internal: str) -> str:
                if settings:
                    return getattr(settings, "resolve_xml_attribute", lambda a, **kw: a)(internal, default=internal) or internal
                return internal
            def _set_attr_teitok(node: ET.Element, internal_attr: str, value: str, default_empty: bool = False) -> None:
                target = _resolve_attr(internal_attr)
                if value and value != "_":
                    node.set(target, value)
                elif default_empty:
                    node.attrib.pop(target, None)
            ord_to_tokid: Dict[int, str] = {}
            n = min(len(tok_elems), len(backend_tokens))
            for i in range(n):
                tok_elem = tok_elems[i]
                token = backend_tokens[i]
                tok_id = tok_elem.get("id") or tok_elem.get("{http://www.w3.org/XML/1998/namespace}id") or f"w-{i+1}"
                if token.id:
                    ord_to_tokid[token.id] = tok_id if tok_id.startswith("w-") else f"w-{i+1}"
            for i in range(n):
                tok_elem = tok_elems[i]
                token = backend_tokens[i]
                tok_elem.set("form", token.form)
                _set_attr_teitok(tok_elem, "lemma", token.lemma or "", default_empty=True)
                _set_attr_teitok(tok_elem, "xpos", token.xpos or "")
                _set_attr_teitok(tok_elem, "upos", token.upos or "")
                _set_attr_teitok(tok_elem, "feats", token.feats or "")
                if token.id:
                    _set_attr_teitok(tok_elem, "ord", str(token.id))
                head_ord = token.head
                if head_ord is not None and head_ord > 0:
                    head_tokid = ord_to_tokid.get(head_ord)
                    if head_tokid:
                        _set_attr_teitok(tok_elem, "head", head_tokid)
                    else:
                        _set_attr_teitok(tok_elem, "head", str(head_ord))
                else:
                    _set_attr_teitok(tok_elem, "head", "", default_empty=True)
                _set_attr_teitok(tok_elem, "deprel", token.deprel or "", default_empty=True)
                if token.misc:
                    _set_attr_teitok(tok_elem, "misc", token.misc)
        # Tokens inside <del> get form="--" (deleted words are not "written"; -- avoids inheriting from innerText, CQP does not index)
        _teitok_set_form_deleted(root)
        # Add revision to TEI header
        change_when = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        backends_used = document.meta.get("_backends_used", []) or ["flexipipe"]
        file_level_attrs = document.meta.get("_file_level_attrs", {})
        model_keys = sorted([k for k in file_level_attrs.keys() if k.endswith("_model")])
        model_str = file_level_attrs[model_keys[0]] if model_keys else None
        change_source = ", ".join(b.upper() for b in backends_used) if len(backends_used) > 1 else (model_str or (backends_used[0].upper() if backends_used else "flexipipe"))
        tasks = set()
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
        tasks_summary_str = ",".join(sorted(tasks)) if tasks else "segment,tokenize"
        change_text = f"Tagged via {change_source} (tasks={tasks_summary_str})"
        _add_change_to_tei_header(root, change_text, change_when, tasks=tasks_summary_str)
        tree = ET.ElementTree(root)
        if HAS_LXML:
            tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True, pretty_print=False)
        else:
            tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True)
        return

    # Parse original XML
    if HAS_LXML:
        parser = ET.XMLParser(strip_cdata=False, remove_blank_text=False)
        tree = ET.parse(str(original_path_obj), parser)
        root = tree.getroot()
    else:
        tree = ET.parse(str(original_path_obj))
        root = tree.getroot()
    
    # Keep a pristine copy of the original XML for later verification
    original_root_snapshot = copy.deepcopy(root)
    
    # Find text nodes
    try:
        text_nodes = root.findall(textnode_xpath.replace(".//", ".//{*}"))
        if not text_nodes:
            text_nodes = root.findall(textnode_xpath)
    except (SyntaxError, ValueError):
        text_nodes = root.findall(textnode_xpath)
    
    if not text_nodes:
        text_nodes = [root]
    
    block_elements_set = {elem.lower() for elem in block_elements}
    extract_elements_set = {elem.lower() for elem in extract_elements}
    
    # Helper functions
    def get_tag_name(elem: ET.Element) -> str:
        return _teitok_local_tag(elem)
    
    def is_extract_element(elem: ET.Element) -> bool:
        return get_tag_name(elem) in extract_elements_set
    
    def is_block_element(elem: ET.Element) -> bool:
        return get_tag_name(elem) in block_elements_set
    
    def is_self_closing_element(elem: ET.Element) -> bool:
        """Check if an element is self-closing (like <lb/>, <pb/>, etc.).
        These elements should never have content - if they do, it's malformed XML."""
        tag_name = get_tag_name(elem)
        self_closing_tags = {'lb', 'pb', 'cb', 'milestone', 'anchor', 'gap', 'fw'}
        return tag_name in self_closing_tags
    
    # Extract content from extract elements (like notes)
    note_content_map: Dict[str, Tuple[str, Dict[str, str], List[ET.Element]]] = {}
    note_id_counter = 1
    
    def extract_note_content(elem: ET.Element) -> Tuple[str, Dict[str, str], List[ET.Element]]:
        text = elem.text or ""
        attrib = dict(elem.attrib)
        children = list(elem)
        return (text, attrib, children)
    
    def extract_all_extract_elements(elem: ET.Element, inside_extract: bool = False) -> None:
        nonlocal note_id_counter
        for child in list(elem):
            child_is_extract = is_extract_element(child)
            if child_is_extract and not inside_extract:
                elem_id = child.get("id") or child.get("{http://www.w3.org/XML/1998/namespace}id")
                if not elem_id:
                    tag_name = get_tag_name(child)
                    elem_id = f"{tag_name}-{note_id_counter}"
                    note_id_counter += 1
                    child.set("id", elem_id)
                note_content_map[elem_id] = extract_note_content(child)
                child.text = None
                for subchild in list(child):
                    child.remove(subchild)
                # Do not recurse into removed content (nested extract elements stay bundled)
                continue
            # Recurse into children (propagate flag if we're inside an extract)
            extract_all_extract_elements(child, inside_extract or child_is_extract)
    
    if not include_notes:
        extract_all_extract_elements(root)
    
    # Track which sentences/tokens have been used
    used_sentence_indices = set()
    used_token_count = 0
    global_tok_id = 1  # Global token ID counter across all block elements
    global_sent_idx = 0  # Global sentence index counter across all block elements
    block_sentence_map: List[List[int]] = document.meta.get("_teitok_block_sentence_map") or []
    block_index = 0
    _align_stats = {"tokenized": 0, "align_skip": 0, "no_sentences": 0, "empty_skip": 0}

    rejoin_meta = get_rejoin_meta(str(original_path_obj)) or document.meta.get("_teitok_rejoin_meta")
    block_display_ranges: List[Tuple[int, int]] = []
    block_nlp_ranges_meta: List[Tuple[int, int]] = []
    global_rejoin_spans: List[RejoinSpan] = []
    if rejoin_meta:
        block_display_ranges = list(rejoin_meta.get("block_display_ranges") or [])
        block_nlp_ranges_meta = list(rejoin_meta.get("block_nlp_ranges") or [])
        global_rejoin_spans = list(rejoin_meta.get("rejoin_spans") or [])
    if block_sentence_map and block_nlp_ranges_meta:
        if len(block_sentence_map) != len(block_nlp_ranges_meta):
            import sys
            print(
                f"\nWARNING: block_sentence_map length {len(block_sentence_map)} != "
                f"block_nlp_ranges {len(block_nlp_ranges_meta)}",
                file=sys.stderr,
            )

    # Build standoff representation for each text node
    for text_node in text_nodes:
        # Find block elements within text node
        block_elems = []
        for child in text_node:
            if is_block_element(child):
                block_elems.append(child)
        # If no direct block children but single child is body/div, use its children (e.g. <p> per paragraph)
        if not block_elems and len(text_node) == 1:
            single = text_node[0]
            if get_tag_name(single) in ('body', 'div'):
                block_elems = [c for c in single if is_block_element(c)]
        if not block_elems:
            # No block elements, process text_node directly
            block_elems = [text_node]
        
        for block_elem in block_elems:
            # Build standoff representation for this block element
            display_plaintext, markup = build_standoff_representation(
                block_elem,
                block_elements_set,
                extract_elements_set,
                include_notes,
                is_self_closing_element  # Pass the function
            )
            block_nlp_range = (
                block_nlp_ranges_meta[block_index]
                if block_index < len(block_nlp_ranges_meta)
                else (0, 0)
            )
            rejoin_spans_block: List[RejoinSpan] = []
            if (
                global_rejoin_spans
                and block_index < len(block_display_ranges)
                and block_index < len(block_nlp_ranges_meta)
            ):
                rejoin_spans_block = slice_rejoin_spans_for_block(
                    block_display_ranges[block_index],
                    block_nlp_range,
                    global_rejoin_spans,
                )
            nlp_full = document.meta.get("_teitok_extracted_nlp", "")
            match_source = "display"
            if nlp_full and block_nlp_range[1] > block_nlp_range[0]:
                match_plaintext = nlp_full[block_nlp_range[0] : block_nlp_range[1]]
                match_source = "nlp_slice"
            elif rejoin_spans_block:
                match_plaintext, rejoin_spans_block = build_nlp_plaintext(
                    display_plaintext, rejoin_spans_block
                )
                match_source = "rejoin_rebuild"
            else:
                match_plaintext = display_plaintext
                match_source = "display"
            plaintext = display_plaintext

            preassigned: Optional[List[int]] = None
            if block_index < len(block_sentence_map) and block_sentence_map[block_index]:
                preassigned = list(block_sentence_map[block_index])
            block_index += 1

            if not display_plaintext.strip() and not match_plaintext.strip():
                _align_stats["empty_skip"] += 1
                continue

            matched_sentences: List[Sentence] = []
            matched_sentence_indices: List[int] = []
            used_preassigned = False

            if preassigned:
                matched_sentences = [document.sentences[i] for i in preassigned]
                matched_sentence_indices = list(preassigned)
                used_preassigned = True
            elif block_nlp_range[1] > block_nlp_range[0]:
                nlp_full = document.meta.get("_teitok_extracted_nlp", "")
                fallback = find_sentences_for_block_nlp_slice(
                    document,
                    nlp_full,
                    block_nlp_range[0],
                    block_nlp_range[1],
                    exclude=used_sentence_indices,
                )
                if fallback:
                    matched_sentences = [document.sentences[i] for i in fallback]
                    matched_sentence_indices = list(fallback)

            fully_consumed_sentence_indices: List[int] = list(matched_sentence_indices)
            nlp_full_for_clip = document.meta.get("_teitok_extracted_nlp", "")
            if (
                matched_sentences
                and nlp_full_for_clip
                and block_nlp_range[1] > block_nlp_range[0]
            ):
                clipped_sents, _clip_sources, fully_consumed_sentence_indices = (
                    clip_sentences_to_nlp_range(
                        document,
                        matched_sentence_indices,
                        nlp_full_for_clip,
                        block_nlp_range[0],
                        block_nlp_range[1],
                    )
                )
                if not clipped_sents:
                    block_id = block_elem.get("id") or block_elem.get(
                        "{http://www.w3.org/XML/1998/namespace}id"
                    )
                    import sys
                    print(
                        f"\nWARNING: No tokens in block NLP slice for block "
                        f"(index={block_index - 1}, id={block_id!r}), skipping...",
                        file=sys.stderr,
                    )
                    _align_stats["no_sentences"] += 1
                    continue
                matched_sentences = clipped_sents

            if not matched_sentences:
                block_id = block_elem.get("id") or block_elem.get(
                    "{http://www.w3.org/XML/1998/namespace}id"
                )
                print(
                    f"\nWARNING: Could not assign sentences for block element "
                    f"(index={block_index - 1}, id={block_id!r}), skipping..."
                )
                _align_stats["no_sentences"] += 1
                continue

            overlap = set(fully_consumed_sentence_indices) & used_sentence_indices
            if overlap:
                print(
                    f"\nWARNING: block {block_index - 1}: sentences {sorted(overlap)} "
                    f"already used; continuing with map assignment",
                    file=__import__("sys").stderr,
                )
            
            # Create a temporary document with matched sentences
            temp_document = Document(id=document.id)
            temp_document.sentences = matched_sentences

            token_positions = map_tokens_to_positions(match_plaintext, temp_document)
            expected_toks = _count_mappable_tokens(temp_document)
            coverage = (len(token_positions) / expected_toks) if expected_toks else 1.0
            if expected_toks and coverage < 0.95:
                block_id = block_elem.get("id") or block_elem.get(
                    "{http://www.w3.org/XML/1998/namespace}id"
                )
                import sys
                print(
                    f"\nWARNING: Token alignment failed for block (index={block_index - 1}, "
                    f"id={block_id!r}, mapped {len(token_positions)}/{expected_toks} tokens), skipping..."
                    + ("" if align_debug else " (use --debug for details)"),
                    file=sys.stderr,
                )
                _align_stats["align_skip"] += 1
                if align_debug:
                    _print_token_align_debug(
                        block_index=block_index - 1,
                        block_id=block_id,
                        block_nlp_range=block_nlp_range,
                        block_display_range=(
                            block_display_ranges[block_index - 1]
                            if block_index - 1 < len(block_display_ranges)
                            else None
                        ),
                        match_source=match_source,
                        display_plaintext=display_plaintext,
                        match_plaintext=match_plaintext,
                        token_positions=token_positions,
                        expected_toks=expected_toks,
                        temp_document=temp_document,
                        preassigned=preassigned,
                        used_preassigned=used_preassigned,
                        nlp_slice_global=(
                            block_nlp_range if match_source == "nlp_slice" else None
                        ),
                    )
                continue

            if rejoin_spans_block:
                token_positions = apply_rejoin_to_token_positions(
                    token_positions, rejoin_spans_block
                )
            token_positions = clamp_token_positions_at_milestones(
                token_positions, markup, plaintext,
                rejoin_spans=rejoin_spans_block if rejoin_spans_block else None,
            )
            sentence_positions = map_sentences_to_positions(plaintext, temp_document, token_positions)
            
            adjusted_markup = _split_markup_for_sentence_boundaries(
                markup,
                sentence_positions,
                block_elements_set,
                extract_elements_set,
                plaintext,
            )
            
            # Rebuild XML with <s> and <tok> elements
            if use_string_rebuild:
                tokens_used, sentences_used = rebuild_xml_with_tokens_string_based(
                    block_elem,
                    plaintext,
                    adjusted_markup,
                    token_positions,
                    sentence_positions,
                    temp_document,
                    block_elements_set,
                    extract_elements_set,
                    note_content_map,
                    start_tok_id=global_tok_id,
                    start_sent_idx=global_sent_idx,
                    settings=settings,
                )
                global_tok_id += tokens_used
                global_sent_idx += sentences_used
            elif use_minidom_rebuild:
                from .insert_tokens_minidom import rebuild_block_xml_minidom
                block_xml_string = ET.tostring(block_elem, encoding="unicode")
                _original_tail = block_elem.tail
                new_xml_string, tokens_used, sentences_used = rebuild_block_xml_minidom(
                    block_xml_string,
                    plaintext,
                    token_positions,
                    sentence_positions,
                    start_tok_id=global_tok_id,
                    start_sent_idx=global_sent_idx,
                )
                if HAS_LXML:
                    new_elem = ET.fromstring(new_xml_string.encode("utf-8"))
                else:
                    new_elem = ET.fromstring(new_xml_string)
                block_elem.clear()
                block_elem.attrib.update(dict(new_elem.attrib))
                for child in list(new_elem):
                    block_elem.append(child)
                if _original_tail is not None:
                    block_elem.tail = _original_tail
                global_tok_id += tokens_used
                global_sent_idx += sentences_used
            else:
                tokens_used, sentences_used = rebuild_xml_with_tokens(
                    block_elem,
                    plaintext,
                    adjusted_markup,
                    token_positions,
                    sentence_positions,
                    temp_document,  # Use temp_document instead of full document
                    block_elements_set,
                    extract_elements_set,
                    note_content_map,
                    start_tok_id=global_tok_id,
                    start_sent_idx=global_sent_idx,
                    settings=settings,
                    rejoin_spans=rejoin_spans_block if rejoin_spans_block else None,
                )
                global_tok_id += tokens_used
                global_sent_idx += sentences_used
            
            # Mark only sentences wholly inside this block (partial overlap stays for next block)
            for sent_idx in fully_consumed_sentence_indices:
                used_sentence_indices.add(sent_idx)
            _align_stats["tokenized"] += 1

    if align_debug:
        import sys
        total_blocks = block_index
        print(
            f"[flexipipe] ALIGN DEBUG summary: blocks_seen={total_blocks} "
            f"tokenized={_align_stats['tokenized']} align_skip={_align_stats['align_skip']} "
            f"no_sentences={_align_stats['no_sentences']} empty_skip={_align_stats['empty_skip']} "
            f"sentences_in_doc={len(document.sentences)}",
            file=sys.stderr,
        )
    
    # Restore note content
    def restore_note_content(root: ET.Element) -> None:
        for elem in root.iter():
            if is_extract_element(elem):
                elem_id = elem.get("id") or elem.get("{http://www.w3.org/XML/1998/namespace}id")
                if elem_id and elem_id in note_content_map:
                    text, attrib, children = note_content_map[elem_id]
                    elem.text = text
                    for key, value in attrib.items():
                        if key != "id":
                            elem.set(key, value)
                    for child in children:
                        elem.append(child)
    
    restore_note_content(root)
    
    # Verify structural integrity (ignoring inserted tags/attributes)
    # If verification fails, raise an exception to prevent writing incorrect XML
    verify_structure_preserved(
        original_root_snapshot,
        root,
        ignore_tags={"s", "tok", "dtok"},  # Also ignore dtok tags (sub-tokens of MWTs)
        ignore_attrs={"id", "rpt"},
        debug=align_debug,
    )
    
    # Add change element to TEI header
    from .teitok import _add_change_to_tei_header
    from datetime import datetime
    change_when = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Build change text from document metadata
    backends_used = document.meta.get("_backends_used", [])
    if not backends_used:
        backends_used = ["flexipipe"]
    
    # Get model information
    file_level_attrs = document.meta.get("_file_level_attrs", {})
    model_keys = sorted([k for k in file_level_attrs.keys() if k.endswith("_model")])
    model_str = None
    if model_keys:
        model_str = file_level_attrs[model_keys[0]]
    
    # Build backend string
    backend_names = [b.upper() for b in backends_used]
    change_source = ", ".join(backend_names) if len(backend_names) > 1 else (model_str or backend_names[0] if backend_names else "flexipipe")
    
    # Detect performed tasks
    tasks = set()
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
    # NER is stored in sentence.entities, not on tokens
    if getattr(document, "spans", None) and document.spans.get("ner"):
        tasks.add("ner")
    elif any(s.entities for s in document.sentences):
        tasks.add("ner")
    # Normalization is detected from various token attributes
    if any(t.reg or t.expan or t.mod or t.trslit or t.ltrslit or t.corr or t.lex or (t.misc and t.misc != "_") for s in document.sentences for t in s.tokens):
        tasks.add("normalize")
    
    tasks_summary_str = ",".join(sorted(tasks)) if tasks else "segment,tokenize"
    change_text = f"Tagged via {change_source} (tasks={tasks_summary_str})"
    
    from .teitok_name_wrap import apply_name_wrappers_to_tree
    apply_name_wrappers_to_tree(root, document)

    _add_change_to_tei_header(root, change_text, change_when, tasks=tasks_summary_str)
    
    # Write updated XML (without pretty-printing)
    if HAS_LXML:
        tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True, pretty_print=False)
    else:
        tree.write(str(output_path_obj), encoding="utf-8", xml_declaration=True)


def verify_structure_preserved(
    original_root: ET.Element,
    modified_root: ET.Element,
    ignore_tags: Set[str],
    ignore_attrs: Set[str],
    *,
    debug: bool = False,
) -> None:
    """
    Compare original vs modified XML by converting both to strings and stripping:
    - Any <s>/<tok> tags (or other ignore_tags)
    - Attributes listed in ignore_attrs (e.g. id, xml:id, rpt)
    - Bridge sequences like </ab><ab rpt="1"> that arise from reopening tags
    """

    def serialize(elem: ET.Element) -> str:
        return ET.tostring(elem, encoding="unicode")

    def remove_tags(text: str, tags_to_remove: Set[str]) -> str:
        if not tags_to_remove:
            return text
        tags_pattern = "|".join(re.escape(tag) for tag in tags_to_remove)
        # Remove opening tags (including attributes) and self-closing forms
        text = re.sub(rf"<(?:{tags_pattern})(\s[^<>]*?)?/?>", "", text, flags=re.IGNORECASE)
        # Remove closing tags
        text = re.sub(rf"</(?:{tags_pattern})>", "", text, flags=re.IGNORECASE)
        return text

    def remove_attributes(text: str, attrs_to_remove: Set[str]) -> str:
        for attr in attrs_to_remove:
            attr_pattern = r'\s+(?:\w+:)?' + re.escape(attr) + r'="[^"]*"'
            text = re.sub(attr_pattern, "", text, flags=re.IGNORECASE)
        return text

    def remove_repetition_bridges(text: str) -> str:
        pattern_whitespace = re.compile(
            r"<([A-Za-z0-9:_-]+)[^<>]*?\brpt=\"1\"[^<>]*?>(\s*)</\1>",
            flags=re.IGNORECASE,
        )
        pattern_bridge = re.compile(
            r"</([A-Za-z0-9:_-]+)>\s*<\1[^<>]*?\brpt=\"1\"[^<>]*?>",
            flags=re.IGNORECASE,
        )
        prev = None
        while prev != text:
            prev = text
            text = pattern_whitespace.sub(r"\2", text)
            text = pattern_bridge.sub("", text)
        return text

    original_text = serialize(original_root)
    modified_text = serialize(modified_root)

    # Apply same normalization steps to both strings
    original_text = remove_tags(original_text, ignore_tags)
    modified_text = remove_tags(modified_text, ignore_tags)

    original_text = remove_repetition_bridges(original_text)
    modified_text = remove_repetition_bridges(modified_text)
    
    original_text = remove_attributes(original_text, ignore_attrs)
    modified_text = remove_attributes(modified_text, ignore_attrs)

    _SELF_CLOSING_COMPARE = {"lb", "pb", "cb", "milestone", "anchor", "gap", "fw"}
    _sc_pattern = (
        r"<(" + "|".join(re.escape(t) for t in _SELF_CLOSING_COMPARE) + r")(\s[^<>]*?)?></\1>"
    )

    def normalize_empty_self_closing(text: str) -> str:
        return re.sub(_sc_pattern, r"<\1\2/>", text, flags=re.IGNORECASE)

    original_text = normalize_empty_self_closing(original_text)
    modified_text = normalize_empty_self_closing(modified_text)

    if original_text != modified_text:
        diff_pos = 0
        max_len = min(len(original_text), len(modified_text))
        while diff_pos < max_len and original_text[diff_pos] == modified_text[diff_pos]:
            diff_pos += 1
        snippet_len = 400 if debug else 120
        snippet_original = original_text[diff_pos:diff_pos + snippet_len]
        snippet_modified = modified_text[diff_pos:diff_pos + snippet_len]
        len_msg = (
            f"\nNormalized lengths: original={len(original_text)} modified={len(modified_text)}"
            if debug
            else ""
        )
        ctx_before = 80 if debug else 0
        before_orig = original_text[max(0, diff_pos - ctx_before):diff_pos]
        before_mod = modified_text[max(0, diff_pos - ctx_before):diff_pos]
        raise ValueError(
            "Modified XML differs from original once token/sentence tags and ignorable attrs are stripped.\n"
            f"First difference at position {diff_pos}:{len_msg}\n"
            + (f"Context before (orig): {before_orig!r}\n" if debug else "")
            + (f"Context before (mod):  {before_mod!r}\n" if debug else "")
            + f"Original: {snippet_original!r}\n"
            + f"Modified: {snippet_modified!r}"
        )
def build_standoff_representation(
    elem: ET.Element,
    block_elements: Set[str],
    extract_elements: Set[str],
    include_notes: bool,
    is_self_closing_element_func
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build plain text and standoff representation from XML element.
    
    Returns:
        plaintext: str - extracted plain text
        markup: List[Dict] - standoff representation with:
            - name: element name
            - contents: attributes and other tag contents
            - start: character position where element starts
            - end: character position where element ends
            - level: nesting depth
            - order: order in XML tree
            - text: optional text at start of element (if not extracted)
            - tail: optional text at end of element (if not extracted)
    """
    plaintext_parts = []
    markup = []
    char_pos = 0
    order = 0
    level = 0
    open_elements = []  # Stack of open elements
    
    def get_tag_name(e: ET.Element) -> str:
        return _teitok_local_tag(e)
    
    def is_extract_element(e: ET.Element) -> bool:
        return get_tag_name(e) in extract_elements
    
    def is_block_element(e: ET.Element) -> bool:
        return get_tag_name(e) in block_elements
    
    def process_element(e: ET.Element) -> None:
        nonlocal char_pos, order, level

        if not isinstance(e.tag, str):
            if e.tail:
                plaintext_parts.append(e.tail)
                char_pos += len(e.tail)
            return
        
        tag_name = get_tag_name(e)
        order += 1
        
        # Build element description
        elem_desc = {
            'name': tag_name,
            'contents': build_element_contents(e),
            'start': char_pos,
            'level': level,
            'order': order,
            'element': e  # Store reference to original element
        }
        
        # Handle extract elements (like notes) - skip their content
        if not include_notes and is_extract_element(e):
            # Extract element: mark position but don't extract content
            # Store the element's text in the markup (not in plaintext)
            def collect_extract_text(elem):
                text_parts = []
                if elem.text:
                    text_parts.append(elem.text)
                for child in elem:
                    text_parts.append(collect_extract_text(child))
                    if child.tail:
                        text_parts.append(child.tail)
                return ''.join(text_parts)
            elem_desc['text'] = collect_extract_text(e)
            elem_desc['end'] = char_pos
            markup.append(elem_desc)
            # Process tail (text after the element) - this SHOULD be extracted
            # The tail is processed by the parent, so we don't process it here
            # Just return - the parent will handle the tail
            return
        
        # Handle self-closing elements (like <lb/>, <pb/>)
        # These elements should never have content - if they do, treat it as tail text
        is_self_closing = is_self_closing_element_func(e)
        if is_self_closing:
            # Self-closing element: open and close at same position
            elem_desc['end'] = char_pos
            # If the element has text content, it's malformed XML - treat it as tail
            if e.text:
                # Store the text as tail (it should be after the element, not inside)
                elem_desc['tail'] = e.text
                # Move the text to tail so parent will process it
                # The text should come before the existing tail (if any)
                e.tail = e.text + (e.tail or "")
                e.text = None
            markup.append(elem_desc)
            # Don't process tail here - let the parent handle it in the child processing loop
            return
        
        # Regular element: extract text and process children
        open_elements.append(elem_desc)
        level += 1
        
        # Process element's text
        if e.text:
            plaintext_parts.append(e.text)
            char_pos += len(e.text)
        
        # Process children
        for child in e:
            process_element(child)
            if child.tail:
                plaintext_parts.append(child.tail)
                char_pos += len(child.tail)
        
        # Close element
        level -= 1
        elem_desc = open_elements.pop()
        elem_desc['end'] = char_pos
        markup.append(elem_desc)
    
    # Process the element
    process_element(elem)
    
    plaintext = ''.join(plaintext_parts)
    return plaintext, markup


def _join_teitok_blocks(block_texts: List[str]) -> str:
    """Join per-block plaintext with \\n\\n separators (matches block offset accounting)."""
    if not block_texts:
        return ""
    out: List[str] = []
    for i, part in enumerate(block_texts):
        if i > 0:
            out.append("\n\n")
        out.append(part)
    return "".join(out)


def extract_plaintext_for_teitok_backend(
    path: str,
    textnode_xpath: str = ".//text",
    include_notes: bool = False,
    block_elements: Optional[List[str]] = None,
    extract_elements: Optional[List[str]] = None,
    *,
    rejoin_linebreaks: bool = True,
    unicode_normalize: Optional[str] = None,
) -> str:
    """
    Extract plain text from TEITOK XML using the same block structure and
    build_standoff_representation logic as insert_tokens_into_teitok.

    This ensures the text sent to the backend is byte-for-byte identical to
    the per-block plaintext we use when matching and rebuilding XML, avoiding
    the 1-char mismatches (e.g. extra space) that come from using a different
    extractor (extract_teitok_plain_text) for loading.
    """
    if block_elements is None:
        block_elements = ['div', 'head', 'p', 'u', 'speaker']
    if extract_elements is None:
        extract_elements = ['note', 'desc', 'gap', 'fw', 'rdg']

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"TEITOK file not found: {path}")

    if HAS_LXML:
        parser = ET.XMLParser(strip_cdata=False, remove_blank_text=False)
        tree = ET.parse(str(path_obj), parser)
        root = tree.getroot()
    else:
        tree = ET.parse(str(path_obj))
        root = tree.getroot()

    block_elements_set = {elem.lower() for elem in block_elements}
    extract_elements_set = {elem.lower() for elem in extract_elements}

    def get_tag_name(elem: ET.Element) -> str:
        return _teitok_local_tag(elem)

    def is_block_element(elem: ET.Element) -> bool:
        return get_tag_name(elem) in block_elements_set

    def is_self_closing_element(elem: ET.Element) -> bool:
        tag_name = get_tag_name(elem)
        self_closing_tags = {'lb', 'pb', 'cb', 'milestone', 'anchor', 'gap', 'fw'}
        return tag_name in self_closing_tags

    try:
        text_nodes = root.findall(textnode_xpath.replace(".//", ".//{*}"))
        if not text_nodes:
            text_nodes = root.findall(textnode_xpath)
    except (SyntaxError, ValueError):
        text_nodes = root.findall(textnode_xpath)

    if not text_nodes:
        text_nodes = [root]

    all_plaintexts: List[str] = []
    all_nlp_plaintexts: List[str] = []
    block_nlp_ranges: List[Tuple[int, int]] = []
    block_display_ranges: List[Tuple[int, int]] = []
    global_rejoin_spans: List[RejoinSpan] = []
    global_offset = 0
    nlp_offset = 0
    normalize_mode = unicode_normalize if unicode_normalize and unicode_normalize != "none" else None
    if normalize_mode:
        from .unicode_utils import normalize_unicode

    for text_node in text_nodes:
        block_elems = [child for child in text_node if is_block_element(child)]
        if not block_elems and len(text_node) == 1:
            single = text_node[0]
            if get_tag_name(single) in ('body', 'div'):
                block_elems = [c for c in single if is_block_element(c)]
        if not block_elems:
            block_elems = [text_node]
        for block_elem in block_elems:
            display_block, markup = build_standoff_representation(
                block_elem,
                block_elements_set,
                extract_elements_set,
                include_notes,
                is_self_closing_element,
            )
            if normalize_mode:
                display_block = normalize_unicode(display_block, normalize_mode) or ""
            block_display_ranges.append((global_offset, global_offset + len(display_block)))
            all_plaintexts.append(display_block)
            if rejoin_linebreaks:
                block_spans = detect_rejoin_spans(display_block, markup)
                if block_spans:
                    nlp_block, block_spans = build_nlp_plaintext(display_block, block_spans)
                    for span in block_spans:
                        global_rejoin_spans.append(
                            RejoinSpan(
                                display_start=span.display_start + global_offset,
                                display_end=span.display_end + global_offset,
                                merged_form=span.merged_form,
                                nlp_start=span.nlp_start + nlp_offset,
                                nlp_end=span.nlp_end + nlp_offset,
                                removed_display_start=span.removed_display_start + global_offset,
                                removed_display_end=span.removed_display_end + global_offset,
                            )
                        )
                else:
                    nlp_block = display_block
                all_nlp_plaintexts.append(nlp_block)
                block_nlp_ranges.append((nlp_offset, nlp_offset + len(nlp_block)))
                nlp_offset += len(nlp_block) + 2
            global_offset += len(display_block) + 2

    display = _join_teitok_blocks(all_plaintexts)
    if not rejoin_linebreaks:
        return display

    nlp_text = _join_teitok_blocks(all_nlp_plaintexts)
    _LAST_REJOIN_META[str(path_obj)] = {
        "display_plaintext": display,
        "nlp_plaintext": nlp_text,
        "rejoin_spans": global_rejoin_spans,
        "block_nlp_ranges": block_nlp_ranges,
        "block_display_ranges": block_display_ranges,
    }
    return nlp_text


_LAST_REJOIN_META: Dict[str, Dict[str, Any]] = {}


def get_rejoin_meta(path: str) -> Optional[Dict[str, Any]]:
    return _LAST_REJOIN_META.get(path)


def store_rejoin_meta(path: str, meta: Dict[str, Any]) -> None:
    _LAST_REJOIN_META[str(path)] = meta


def count_teitok_blocks_in_xml(
    path: str,
    textnode_xpath: str = ".//text",
    block_elements: Optional[List[str]] = None,
) -> int:
    """Count block elements the same way extract/writeback iterate them."""
    if block_elements is None:
        block_elements = ["div", "head", "p", "u", "speaker"]
    block_elements_set = {e.lower() for e in block_elements}
    path_obj = Path(path)
    if HAS_LXML:
        parser = ET.XMLParser(strip_cdata=False, remove_blank_text=False)
        root = ET.parse(str(path_obj), parser).getroot()
    else:
        root = ET.parse(str(path_obj)).getroot()
    try:
        text_nodes = root.findall(textnode_xpath.replace(".//", ".//{*}"))
        if not text_nodes:
            text_nodes = root.findall(textnode_xpath)
    except (SyntaxError, ValueError):
        text_nodes = root.findall(textnode_xpath)
    if not text_nodes:
        text_nodes = [root]

    def is_block(elem: ET.Element) -> bool:
        tag = _teitok_local_tag(elem)
        return tag in block_elements_set

    count = 0
    for text_node in text_nodes:
        block_elems = [c for c in text_node if is_block(c)]
        if not block_elems and len(text_node) == 1:
            single = text_node[0]
            if _teitok_local_tag(single) in ("body", "div"):
                block_elems = list(single)
        if not block_elems:
            block_elems = [text_node]
        count += len(block_elems)
    return count


def _split_markup_for_sentence_boundaries(
    markup: List[Dict[str, Any]],
    sentence_positions: List[Tuple[int, int, Sentence]],
    block_elements: Set[str],
    extract_elements: Set[str],
    plaintext: str,
) -> List[Dict[str, Any]]:
    """
    Split markup spans so that sentence boundaries always align with start/end positions
    for splittable tags (prevents unnecessary fallbacks).
    """
    if not sentence_positions:
        return markup

    split_positions: Set[int] = set()
    text_len = len(plaintext)
    for _, end, _ in sentence_positions:
        adjusted_end = end
        while adjusted_end < text_len and plaintext[adjusted_end].isspace():
            adjusted_end += 1
        split_positions.add(adjusted_end)

    if not split_positions:
        return markup

    # Do not split inline/mixed-content elements at sentence boundaries; splitting
    # would change structure (e.g. one <hi> becomes </hi><hi>) and fail structure verification.
    inline_never_break = {'hi', 'emph', 'b', 'i', 'u', 'span', 'ref', 'name', 'lb', 'pb'}
    never_break = set(block_elements) | set(extract_elements) | inline_never_break
    result: List[Dict[str, Any]] = []

    for entry in markup:
        tag_name = entry.get("name")
        start = entry.get("start")
        end = entry.get("end")
        tag_lower = (tag_name or "").lower() if isinstance(tag_name, str) else ""
        # Never split when tag name is missing or empty (preserve structure)
        if not tag_lower:
            result.append(entry)
            continue

        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start >= end
            or tag_lower in never_break
        ):
            result.append(entry)
            continue

        splits = sorted(pos for pos in split_positions if start < pos < end)
        if not splits:
            result.append(entry)
            continue

        boundaries = [start] + splits + [end]
        nonempty_segments: List[Tuple[int, int]] = []
        current_start = boundaries[0]
        for idx in range(1, len(boundaries)):
            seg_start = current_start
            seg_end = boundaries[idx]
            segment_text = plaintext[seg_start:seg_end]
            if segment_text.strip():
                nonempty_segments.append((seg_start, seg_end))
                current_start = seg_end
            else:
                current_start = seg_end
                if nonempty_segments:
                    last_start, _ = nonempty_segments[-1]
                    nonempty_segments[-1] = (last_start, seg_end)
        for idx, (seg_start, seg_end) in enumerate(nonempty_segments):
            segment_entry = dict(entry)
            segment_entry["start"] = seg_start
            segment_entry["end"] = seg_end
            if idx > 0:
                segment_entry["split_reopen"] = True
            else:
                segment_entry.pop("split_reopen", None)
            result.append(segment_entry)

    return result


def build_element_contents(elem: ET.Element) -> str:
    """Build the contents string for an element (attributes, etc.)."""
    parts = []
    for key, value in elem.attrib.items():
        # Handle namespace prefixes
        if '}' in key:
            key = key.split('}')[1]
        parts.append(f'{key}="{value}"')
    return ' ' + ' '.join(parts) if parts else ''


def _mappable_token_forms(document: Document) -> List[str]:
    """Return surface forms in map_tokens_to_positions walk order (skip MWT subtokens)."""
    subtoken_ids: Set[int] = set()
    for sent in document.sentences:
        for token in sent.tokens:
            if token.is_mwt and token.subtokens:
                for st in token.subtokens[1:]:
                    if st.id:
                        subtoken_ids.add(st.id)
    forms: List[str] = []
    for sent in document.sentences:
        for token in sent.tokens:
            if token.id and token.id in subtoken_ids:
                continue
            forms.append(token.form)
    return forms


def _diagnose_token_alignment(
    plaintext: str,
    document: Document,
) -> Dict[str, Any]:
    """
    Mirror map_tokens_to_positions; report first token that cannot be placed.
    """
    subtoken_ids: Set[int] = set()
    for sent in document.sentences:
        for token in sent.tokens:
            if token.is_mwt and token.subtokens:
                for st in token.subtokens[1:]:
                    if st.id:
                        subtoken_ids.add(st.id)

    text_pos = 0
    mapped = 0
    token_ordinal = 0
    first_failure: Optional[Dict[str, Any]] = None
    failures: List[Dict[str, Any]] = []

    for sent_idx, sent in enumerate(document.sentences):
        for tok_idx, token in enumerate(sent.tokens):
            if token.id and token.id in subtoken_ids:
                continue
            token_ordinal += 1
            while text_pos < len(plaintext) and plaintext[text_pos].isspace():
                text_pos += 1
            if text_pos >= len(plaintext):
                info = {
                    "reason": "plaintext_exhausted",
                    "text_pos": text_pos,
                    "plaintext_len": len(plaintext),
                    "sent_idx": sent_idx,
                    "tok_idx": tok_idx,
                    "token_id": token.id,
                    "form": token.form,
                    "token_ordinal": token_ordinal,
                }
                failures.append(info)
                if first_failure is None:
                    first_failure = info
                continue
            tok_form = token.form
            if plaintext[text_pos:].startswith(tok_form):
                text_pos += len(tok_form)
                mapped += 1
            elif plaintext[text_pos:].lower().startswith(tok_form.lower()):
                text_pos += len(tok_form)
                mapped += 1
            else:
                remaining = plaintext[text_pos:]
                match = re.search(re.escape(tok_form), remaining, re.IGNORECASE)
                if match:
                    text_pos += match.end()
                    mapped += 1
                else:
                    ctx_end = min(len(plaintext), text_pos + 60)
                    info = {
                        "reason": "form_not_found",
                        "text_pos": text_pos,
                        "plaintext_len": len(plaintext),
                        "sent_idx": sent_idx,
                        "tok_idx": tok_idx,
                        "token_id": token.id,
                        "form": tok_form,
                        "token_ordinal": token_ordinal,
                        "at_pos": repr(plaintext[text_pos:ctx_end]),
                        "search_remaining_len": len(remaining),
                    }
                    failures.append(info)
                    if first_failure is None:
                        first_failure = info

    tail_start = text_pos
    while tail_start < len(plaintext) and plaintext[tail_start].isspace():
        tail_start += 1
    unconsumed = plaintext[tail_start:] if tail_start < len(plaintext) else ""

    return {
        "mapped": mapped,
        "expected": _count_mappable_tokens(document),
        "first_failure": first_failure,
        "failures": failures[:12],
        "unconsumed_tail": unconsumed,
        "token_forms": _mappable_token_forms(document),
    }


def _snippet(text: str, max_len: int = 240) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _print_token_align_debug(
    *,
    block_index: int,
    block_id: Optional[str],
    block_nlp_range: Tuple[int, int],
    block_display_range: Optional[Tuple[int, int]],
    match_source: str,
    display_plaintext: str,
    match_plaintext: str,
    token_positions: List[Tuple[int, int, Token]],
    expected_toks: int,
    temp_document: Document,
    preassigned: Optional[List[int]],
    used_preassigned: bool,
    nlp_slice_global: Optional[Tuple[int, int]],
) -> None:
    import sys

    diag = _diagnose_token_alignment(match_plaintext, temp_document)
    forms = diag.get("token_forms") or []
    joined = " ".join(forms)
    clipped_toks = _count_mappable_tokens(temp_document)
    lines = [
        f"[flexipipe] ALIGN DEBUG block index={block_index} id={block_id!r}",
        f"  match_source={match_source} block_nlp_range={block_nlp_range} "
        f"block_display_range={block_display_range} nlp_slice_global={nlp_slice_global}",
        f"  display_len={len(display_plaintext)} match_len={len(match_plaintext)} "
        f"mapped={len(token_positions)}/{expected_toks} diagnose_mapped={diag.get('mapped')}/{diag.get('expected')} "
        f"clipped_mappable_tokens={clipped_toks}",
        f"  preassigned={preassigned} used_preassigned={used_preassigned}",
        f"  display_plaintext[{_snippet(display_plaintext)}]",
        f"  match_plaintext[{_snippet(match_plaintext)}]",
        f"  token_forms_joined[{_snippet(joined)}]",
    ]
    if display_plaintext != match_plaintext:
        lines.append(
            f"  display!=match (first diff may matter for rebuild; align uses match_plaintext)"
        )
    ff = diag.get("first_failure")
    if ff:
        lines.append(f"  first_failure: {ff}")
    for extra in diag.get("failures") or []:
        if extra is not ff:
            lines.append(f"  also_failed: {extra}")
    tail = diag.get("unconsumed_tail") or ""
    if tail.strip():
        lines.append(f"  unconsumed_match_tail[{_snippet(tail)}]")
    if forms and match_plaintext:
        # Character-level prefix comparison (first diverging token)
        pos = 0
        for i, form in enumerate(forms[:6]):
            while pos < len(match_plaintext) and match_plaintext[pos].isspace():
                pos += 1
            if pos >= len(match_plaintext):
                lines.append(f"  prefix_check: token[{i}] form={form!r} but match_plaintext ended at {pos}")
                break
            if not match_plaintext[pos:].startswith(form):
                lines.append(
                    f"  prefix_check: token[{i}] form={form!r} expected at pos={pos} "
                    f"got={match_plaintext[pos:pos + min(40, len(match_plaintext) - pos)]!r}"
                )
                break
            pos += len(form)
    print("\n".join(lines), file=sys.stderr)


def _count_mappable_tokens(document: Document) -> int:
    subtoken_ids: Set[int] = set()
    for sent in document.sentences:
        for token in sent.tokens:
            if token.is_mwt and token.subtokens:
                for st in token.subtokens[1:]:
                    if st.id:
                        subtoken_ids.add(st.id)
    count = 0
    for sent in document.sentences:
        for token in sent.tokens:
            if token.id and token.id in subtoken_ids:
                continue
            count += 1
    return count


_SELF_CLOSING_MARKUP_TAGS = frozenset(
    {"lb", "pb", "cb", "milestone", "anchor", "gap", "fw"}
)


def _token_covers_rejoin_span(
    tok_start: int,
    tok_end: int,
    rejoin_spans: Optional[List[RejoinSpan]],
) -> bool:
    """True if the token span fully contains a line-break rejoin span.

    Such a token is *meant* to straddle <lb/>/<pc/> markup: the word was split
    across a line break and the merged token must keep that inline markup inside
    the <tok> element. Clamping it away from the milestone would both drop the
    rejoin and corrupt the surrounding text.
    """
    if not rejoin_spans:
        return False
    for span in rejoin_spans:
        if span.display_end <= span.display_start:
            continue
        if tok_start <= span.display_start and span.display_end <= tok_end:
            return True
    return False


def _mwt_surface_is_plain(
    tok_start: int,
    tok_end: int,
    token: Token,
    plaintext: str,
    markup: List[Dict[str, Any]],
    block_elem: ET.Element,
) -> bool:
    """True if an MWT may be rendered with <tok> text taken straight from its form.

    That shortcut (``tok_elem.text = token.form``) is only safe when the token's
    display span is a contiguous run of text that exactly equals the combined
    form. If inline markup (<lb/>, <pc/>, ...) falls inside the span, or the span
    no longer matches the form (e.g. it was clamped at a milestone), the surface
    must be rebuilt character-by-character so the original markup is preserved
    rather than dropped, and the pre-markup text is not duplicated.
    """
    if plaintext[tok_start:tok_end] != (token.form or ""):
        return False
    for m in markup:
        if m.get("element") is block_elem:
            continue
        ms = m.get("start")
        if isinstance(ms, int) and tok_start < ms < tok_end:
            return False
    return True


def clamp_token_positions_at_milestones(
    token_positions: List[Tuple[int, int, Token]],
    markup: List[Dict[str, Any]],
    plaintext: str,
    rejoin_spans: Optional[List[RejoinSpan]] = None,
) -> List[Tuple[int, int, Token]]:
    """Keep <lb/>/<cb/> outside token spans when rejoin mapping overlaps a milestone.

    Tokens that legitimately span a milestone -- line-break rejoin spans and
    multi-word tokens -- are left untouched so that the <lb/>/<pc/> markup of a
    hyphenated, line-broken word stays inside the merged <tok> element.
    """
    milestones = sorted(
        m["start"]
        for m in markup
        if m.get("name") in _SELF_CLOSING_MARKUP_TAGS
        and m.get("start") == m.get("end")
    )
    if not milestones:
        return token_positions
    out: List[Tuple[int, int, Token]] = []
    for s, e, tok in token_positions:
        # Tokens that legitimately span markup -- line-break rejoins and
        # multi-word tokens -- must keep the <lb/>/<pc/> inside the <tok>.
        # The rebuild reconstructs their surface from the original text and
        # markup, so clamping them here would only drop or duplicate text.
        if getattr(tok, "is_mwt", False) or _token_covers_rejoin_span(
            s, e, rejoin_spans
        ):
            out.append((s, e, tok))
            continue
        # A token whose display span already matches its form was aligned to the
        # full surface (e.g. kwašení<lb/>národů → form kwašenínárodů). Zero-width
        # milestones inside that span must stay within the <tok>, not be clamped
        # away — which would leave the pre-<lb/> text bare in the XML.
        form = tok.form or ""
        if form and plaintext[s:e] == form:
            out.append((s, e, tok))
            continue
        ns, ne = s, e
        form = tok.form or ""
        for z in milestones:
            if ns < z < ne:
                suffix = plaintext[z:ne]
                if form and suffix.startswith(form):
                    ns = z
                elif form and plaintext[ns:z].strip().endswith(form):
                    ne = z
                else:
                    ns = max(ns, z)
        if ne > ns:
            out.append((ns, ne, tok))
    return out


_LOCAL_TOKEN_SEARCH_WINDOW = 48


def _token_form(token: Token) -> str:
    return token.form or ""


def _map_token_at_cursor(
    plaintext: str, token: Token, cursor: int
) -> Optional[Tuple[int, int]]:
    """Map one token at or shortly after cursor (never jump far ahead in the block)."""
    form = _token_form(token)
    if not form:
        return None
    pos = cursor
    while pos < len(plaintext) and plaintext[pos].isspace():
        pos += 1
    if pos >= len(plaintext):
        return None
    if plaintext[pos : pos + len(form)] == form:
        return pos, pos + len(form)
    tail = plaintext[pos : pos + _LOCAL_TOKEN_SEARCH_WINDOW]
    lower_tail = tail.lower()
    lower_form = form.lower()
    if lower_tail.startswith(lower_form):
        return pos, pos + len(form)
    rel = lower_tail.find(lower_form)
    if rel >= 0:
        return pos + rel, pos + rel + len(form)
    return None


def map_tokens_to_positions(plaintext: str, document: Document) -> List[Tuple[int, int, Token]]:
    """
    Map tokens to character positions in plain text.

    Each sentence is anchored with find_tokens_at so common short forms (e.g. "a", "se")
    are not matched far ahead via a global regex search, which left bare words between
    tokens and between </s> boundaries.
    """
    subtoken_ids: Set[int] = set()
    for sent in document.sentences:
        for token in sent.tokens:
            if token.is_mwt and token.subtokens:
                for st in token.subtokens[1:]:
                    if st.id:
                        subtoken_ids.add(st.id)

    positions: List[Tuple[int, int, Token]] = []
    search_from = 0

    for sent in document.sentences:
        tokens = _sentence_tokens(sent)
        if not tokens:
            continue

        anchor = find_tokens_at(plaintext, tokens, search_from)
        cursor = anchor if anchor >= 0 else search_from

        for token in tokens:
            mapped = _map_token_at_cursor(plaintext, token, cursor)
            if mapped is None:
                continue
            start_pos, end_pos = mapped
            positions.append((start_pos, end_pos, token))
            cursor = end_pos

        search_from = max(search_from, cursor)

    positions = _recover_missing_token_positions(
        plaintext, positions, document, subtoken_ids
    )
    return positions


def _recover_missing_token_positions(
    plaintext: str,
    positions: List[Tuple[int, int, Token]],
    document: Document,
    subtoken_ids: Set[int],
) -> List[Tuple[int, int, Token]]:
    """Place tokens that map_tokens skipped, searching only between neighbors."""
    pos_by_id: Dict[int, Tuple[int, int, Token]] = {}
    for s, e, tok in positions:
        if tok.id:
            pos_by_id[tok.id] = (s, e, tok)

    recovered = list(positions)
    ordered: List[Token] = []
    for sent in document.sentences:
        for token in sent.tokens:
            if token.id and token.id in subtoken_ids:
                continue
            ordered.append(token)

    for i, token in enumerate(ordered):
        if token.id and token.id in pos_by_id:
            continue
        form = _token_form(token)
        if not form:
            continue
        prev_end = 0
        for j in range(i - 1, -1, -1):
            prev = ordered[j]
            if prev.id and prev.id in pos_by_id:
                prev_end = pos_by_id[prev.id][1]
                break
        next_start = len(plaintext)
        for j in range(i + 1, len(ordered)):
            nxt = ordered[j]
            if nxt.id and nxt.id in pos_by_id:
                next_start = pos_by_id[nxt.id][0]
                break
        if next_start <= prev_end:
            continue
        chunk = plaintext[prev_end:next_start]
        idx = chunk.find(form)
        if idx < 0:
            idx = chunk.lower().find(form.lower())
        if idx < 0:
            continue
        start = prev_end + idx
        end = start + len(form)
        if any(not (end <= s or start >= e) for s, e, _ in recovered):
            continue
        recovered.append((start, end, token))
        if token.id:
            pos_by_id[token.id] = (start, end, token)

    recovered.sort(key=lambda x: x[0])
    return recovered


def map_sentences_to_positions(
    plaintext: str, 
    document: Document, 
    token_positions: List[Tuple[int, int, Token]]
) -> List[Tuple[int, int, Sentence]]:
    """
    Map sentences to character positions based on token positions.
    
    Ensures sentences always span at least from the first token's start to the last token's end.
    If sentence text is available, extends to include trailing whitespace/punctuation.
    Ensures sentences don't overlap - each sentence ends before the next one starts.
    
    Returns:
        List of (start_pos, end_pos, sentence) tuples
    """
    positions = []
    token_idx = 0
    
    for i, sent in enumerate(document.sentences):
        if token_idx >= len(token_positions):
            break
        
        first_token_start = token_positions[token_idx][0]
        
        # Find the last token that belongs to this sentence
        # We need to match tokens from token_positions to tokens in sent.tokens
        # MWT subtokens are skipped in token_positions, so we need to account for that
        last_token_idx = token_idx
        sent_token_idx = 0
        pos_idx = token_idx
        
        # Iterate through tokens in token_positions, matching them to sent.tokens
        while pos_idx < len(token_positions) and sent_token_idx < len(sent.tokens):
            token_from_pos = token_positions[pos_idx][2]  # Token object from positions
            sent_token = sent.tokens[sent_token_idx]
            
            # Check if this is the same token
            if token_from_pos == sent_token or (token_from_pos.id and sent_token.id and token_from_pos.id == sent_token.id):
                # Match found - this token belongs to this sentence
                last_token_idx = pos_idx
                pos_idx += 1
                sent_token_idx += 1
            else:
                # This sent_token is likely a subtoken (not in token_positions)
                # Skip it and continue
                sent_token_idx += 1
        
        # Calculate last token end position
        if last_token_idx >= len(token_positions):
            # Fallback: use first token position if we can't find last token
            last_token_end = first_token_start
        else:
            last_token_end = token_positions[last_token_idx][1]
        
        # Determine the maximum end position - don't go beyond next sentence's start
        # This is the STRICT upper bound - sentence must end before next sentence starts
        max_sentence_end = len(plaintext)
        if i + 1 < len(document.sentences):
            # Find where the next sentence starts
            # The next sentence starts at the token after the last token of this sentence
            next_token_idx = last_token_idx + 1
            if next_token_idx < len(token_positions):
                next_sentence_start = token_positions[next_token_idx][0]
                # CRITICAL: Ensure this sentence ends BEFORE the next one starts
                # This prevents overlap - use the next sentence's start as the hard limit
                max_sentence_end = next_sentence_start
            else:
                # No more tokens - also try to find next sentence by its text
                next_sent = document.sentences[i + 1]
                if next_sent.text:
                    next_sent_text = next_sent.text.strip()
                    # Search for next sentence's text in plaintext starting from last_token_end
                    import re
                    next_sent_pattern = re.escape(next_sent_text)
                    next_sent_pattern = next_sent_pattern.replace(r'\ ', r'\s+')
                    match = re.search(next_sent_pattern, plaintext[last_token_end:], re.IGNORECASE)
                    if match:
                        next_sentence_start = last_token_end + match.start()
                        max_sentence_end = next_sentence_start
        
        # Sentence span covers mapped tokens plus trailing whitespace only (never bare
        # words between the last </tok> and </s>, nor between </s> and the next <s>).
        sentence_end = last_token_end
        while (
            sentence_end < max_sentence_end
            and sentence_end < len(plaintext)
            and plaintext[sentence_end].isspace()
        ):
            sentence_end += 1
        
        # FINAL CHECK: Ensure sentence doesn't extend beyond max_sentence_end (next sentence's start)
        # This is a hard limit - never exceed it
        sentence_end = min(sentence_end, max_sentence_end)
        
        # Ensure sentence doesn't extend beyond plaintext
        sentence_end = min(sentence_end, len(plaintext))
        
        # CRITICAL: Final check - ensure sentence doesn't extend beyond max_sentence_end
        # This is already set above based on next sentence's start, but double-check here
        if sentence_end > max_sentence_end:
            sentence_end = max_sentence_end
        
        # Ensure this sentence doesn't overlap with the previous sentence
        if positions:
            prev_start, prev_end, _ = positions[-1]
            if first_token_start < prev_end:
                # This sentence starts before the previous one ends - this shouldn't happen
                # but if it does, adjust the previous sentence's end to be before this one starts
                # Actually, we can't modify previous positions, so ensure this one starts after previous ends
                first_token_start = max(first_token_start, prev_end)
                sentence_end = max(sentence_end, first_token_start)
            # Also ensure this sentence ends before it would overlap with previous
            if sentence_end <= prev_end and first_token_start < prev_end:
                # This sentence is completely within previous - shouldn't happen, but adjust
                sentence_end = prev_end + 1
        
        # Final check: ensure sentence_end > first_token_start (sentence must have positive length)
        if sentence_end <= first_token_start:
            sentence_end = first_token_start + 1
        
        positions.append((first_token_start, sentence_end, sent))
        # Update token_idx to point to the first token of the next sentence
        # Use last_token_idx + 1 (the token after the last token of this sentence)
        token_idx = last_token_idx + 1
    
    return positions


def rebuild_xml_with_tokens(
    block_elem: ET.Element,
    plaintext: str,
    markup: List[Dict[str, Any]],
    token_positions: List[Tuple[int, int, Token]],
    sentence_positions: List[Tuple[int, int, Sentence]],
    document: Document,
    block_elements: Set[str],
    extract_elements: Set[str],
    note_content_map: Dict[str, Tuple[str, Dict[str, str], List[ET.Element]]],
    start_tok_id: int = 1,
    start_sent_idx: int = 0,
    settings: Optional[Any] = None,
    rejoin_spans: Optional[List[RejoinSpan]] = None,
) -> Tuple[int, int]:
    """
    Rebuild XML by inserting <s> and <tok> elements at correct positions.
    
    This is the core function that uses the standoff representation to correctly
    insert tokens and sentences while preserving all original XML structure.
    """
    # Helper functions for attribute mapping (similar to teitok.py)
    def _resolve_attr_name(internal_attr: str) -> Optional[str]:
        if settings:
            return settings.resolve_xml_attribute(internal_attr, default=internal_attr)
        return internal_attr
    
    def _attribute_aliases(internal_attr: str) -> List[str]:
        if settings:
            aliases = settings.get_attribute_mapping(internal_attr)
        else:
            aliases = [internal_attr]
        ordered: List[str] = []
        for alias in aliases + [internal_attr]:
            if alias and alias not in ordered:
                ordered.append(alias)
        return ordered
    
    def _set_attr(node: ET.Element, internal_attr: str, value: str, default_empty: bool = False) -> None:
        """Set attribute respecting TEITOK mappings."""
        target_attr = _resolve_attr_name(internal_attr)
        aliases = _attribute_aliases(internal_attr)
        
        if target_attr is None:
            for alias in aliases:
                node.attrib.pop(alias, None)
            return
        if value and value != "_":
            node.set(target_attr, value)
            for alias in aliases:
                if alias != target_attr:
                    node.attrib.pop(alias, None)
        elif default_empty:
            for alias in aliases:
                node.attrib.pop(alias, None)
        elif target_attr != internal_attr:
            for alias in aliases:
                if alias != target_attr:
                    node.attrib.pop(alias, None)
    
    # Save original attributes
    original_attrs = dict(block_elem.attrib)
    original_tail = block_elem.tail
    block_elem.clear()
    block_elem.attrib.update(original_attrs)
    
    # Build maps for quick lookup
    # Map character position to list of elements that open/close there
    opens_at = {}  # pos -> list of (type, data) where type is 'markup', 'sentence', or 'token'
    closes_at = {}  # pos -> list of (type, data)
    # Map character position to the element whose tail it belongs to
    char_to_tail_elem = {}  # char_pos -> (element, start_pos, end_pos)
    
    for m in markup:
        start = m['start']
        end = m['end']
        if start not in opens_at:
            opens_at[start] = []
        opens_at[start].append(('markup', m))
        if end not in closes_at:
            closes_at[end] = []
        closes_at[end].append(('markup', m))
        
        # Track tail text for self-closing elements
        # For self-closing elements, the tail comes AFTER the element
        # The element's end position is where it "closes" (same as start for self-closing)
        # The tail starts at end and continues for len(tail) characters
        if 'tail' in m and m['tail']:
            tail_start = end
            tail_end = end + len(m['tail'])
            for tail_pos in range(tail_start, tail_end):
                char_to_tail_elem[tail_pos] = (m, tail_start, tail_end)
    
    # Add sentences and tokens to opens_at/closes_at
    for sent_start, sent_end, sent in sentence_positions:
        if sent_start not in opens_at:
            opens_at[sent_start] = []
        opens_at[sent_start].append(('sentence', (sent_start, sent_end, sent)))
        if sent_end not in closes_at:
            closes_at[sent_end] = []
        closes_at[sent_end].append(('sentence', (sent_start, sent_end, sent)))
    
    for tok_start, tok_end, token in token_positions:
        if tok_start not in opens_at:
            opens_at[tok_start] = []
        opens_at[tok_start].append(('token', (tok_start, tok_end, token)))
        if tok_end not in closes_at:
            closes_at[tok_end] = []
        closes_at[tok_end].append(('token', (tok_start, tok_end, token)))
    
    def compute_sentence_fallbacks(sentence_positions, markup, text):
        fallback = set()
        for idx, (s_start, s_end, _) in enumerate(sentence_positions):
            for m in markup:
                if m['element'] == block_elem or m['start'] == m['end']:
                    continue
                m_start = m['start']
                m_end = m['end']
                overlap_before = m_start < s_start < m_end and text[m_start:s_start].strip()
                overlap_after = m_start < s_end < m_end and text[s_end:m_end].strip()
                if overlap_before or overlap_after:
                    fallback.add(idx)
                    break
        return fallback
    
    sentence_fallbacks = compute_sentence_fallbacks(sentence_positions, markup, plaintext)
    
    # Note: Implicit MWT detection is handled in __main__.py via _create_implicit_mwt
    # before calling insert_tokens_into_teitok, so we don't need to do it here.
    # This avoids conflicts and ensures MWTs are created consistently.
    
    # Track token IDs that are subtokens of MWT tokens (these should be skipped during insertion)
    mwt_subtoken_ids = set()
    for sent_start, sent_end, sent in sentence_positions:
        sent_tokens = [(t_start, t_end, tok) for t_start, t_end, tok in token_positions 
                      if sent_start <= t_start < sent_end]
        sent_tokens.sort(key=lambda x: x[0])
        for t_start, t_end, tok in sent_tokens:
            if tok.is_mwt and tok.subtokens:
                # Mark all subtokens (except the first one, which is tok itself) as MWT subtokens to skip
                for st in tok.subtokens[1:]:
                    # Track by token ID instead of Token object (Token objects are not hashable)
                    if st.id:
                        mwt_subtoken_ids.add(st.id)
    
    # Track current state
    current_elem = block_elem
    open_markup_stack = []  # Stack of (markup_entry, element) tuples
    token_idx = 0
    sent_idx = start_sent_idx  # Start from the provided index
    global_tok_id = start_tok_id  # Start from the provided token ID
    current_tok_elem = None
    current_sent_elem = None
    current_sentence_idx = None
    current_sentence_is_fallback = False
    current_sentence_first_token = None
    current_sentence_obj = None
    sent_token_ids = []
    # Map from token ord (id) to tokid for head conversion
    ord_to_tokid: Dict[int, str] = {}
    # Track self-closing elements that were just inserted (for tail text handling)
    self_closing_elements = {}  # (start_pos, end_pos) -> element
    # Track token state when markup elements open while a token is active
    markup_token_state: Dict[ET.Element, Dict[str, Any]] = {}
    # When closing markup with token spanning boundary, we may inject plaintext into reopened elem;
    # skip adding those characters in the character loop.
    skip_char_until: Optional[int] = None
    # Build parent map for xml.etree.ElementTree compatibility
    parent_map = {c: p for p in block_elem.iter() for c in p}
    
    def resolve_parent_after_token(parent):
        candidate = parent
        open_elements = {elem for _, elem in open_markup_stack}
        while candidate is not None:
            if current_sent_elem is not None and candidate == current_sent_elem:
                return candidate
            if candidate in open_elements:
                return candidate
            candidate = parent_map.get(candidate)
        return block_elem
    
    # Process each character position
    initial_tok_id = start_tok_id
    initial_sent_idx = start_sent_idx  # Use the provided start index
    for char_pos in range(len(plaintext) + 1):  # +1 to handle closing at end
        char = plaintext[char_pos] if char_pos < len(plaintext) else None
        tail_elem = None  # Element whose tail this char belongs to (if any), reset each iteration

        # Close elements that end at this position (before opening new ones)
        # Close in reverse order: markup first (innermost), then tokens, then sentences (outermost)
        if char_pos in closes_at:
            # Sort by type: markup first (innermost), then tokens, then sentences (outermost)
            # Within each type, close those that end later first (to close innermost first)
            def get_close_key(item):
                item_type, item_data = item
                if item_type == 'markup':
                    level = item_data.get('level', 0)
                    return (0, -item_data['end'], -level)  # Markup first, deeper first
                elif item_type == 'token':
                    return (1, -item_data[1])  # Tokens second, later end first
                else:  # sentence
                    return (2, -item_data[1])  # Sentences last, later end first
            
            to_close = sorted(closes_at[char_pos], key=get_close_key)
            
            for close_type, close_data in to_close:
                if close_type == 'sentence':
                    sent_start, sent_end, sent = close_data
                    if char_pos == sent_end and current_sent_elem is not None:
                        # Close sentence
                        if not current_sent_elem.text and not list(current_sent_elem) and sent_token_ids:
                            # Empty sentence with tokens - use @corresp
                            current_sent_elem.set("corresp", " ".join(f"#{tid}" for tid in sent_token_ids))
                        parent_after_sentence = parent_map.get(current_sent_elem, block_elem)
                        if open_markup_stack:
                            current_elem = open_markup_stack[-1][1]
                        else:
                            current_elem = parent_after_sentence
                        current_sent_elem = None
                        sent_token_ids = []
                        sent_idx += 1
                elif close_type == 'token':
                    tok_start, tok_end, token = close_data
                    if char_pos == tok_end and current_tok_elem is not None:
                        # Close token
                        # Clear any markup state tied to this token
                        for elem in list(markup_token_state.keys()):
                            if markup_token_state[elem].get("token") == current_tok_elem:
                                markup_token_state.pop(elem, None)
                        parent_after_token = parent_map.get(current_tok_elem, block_elem)
                        current_elem = resolve_parent_after_token(parent_after_token)
                        current_tok_elem = None
                        token_idx += 1
                elif close_type == 'markup':
                    m = close_data
                    # Find this markup in the open stack and close it
                    # Compare by start/end positions and tag name
                    for i, (open_m, open_elem) in enumerate(open_markup_stack):
                        # Match by start/end positions and tag name
                        if (open_m['start'] == m['start'] and 
                            open_m['end'] == m['end'] and
                            open_m['name'] == m['name']):
                            # Close this element
                            # If a split continuation (another segment with same tag opening at this position)
                            # exists in markup, do a simple close so opens_at will open it and characters go to it.
                            m_name = (m.get('name') or '').lower() if isinstance(m.get('name'), str) else ''
                            has_split_continuation = bool(
                                m_name and any(
                                    m2 is not m
                                    and (m2.get('name') or '').lower() == m_name
                                    and m2.get('start') == char_pos
                                    and m2.get('end', 0) > char_pos
                                    for m2 in markup
                                )
                            )
                            if has_split_continuation:
                                markup_token_state.pop(open_elem, None)
                                current_elem = parent_map.get(open_elem, block_elem)
                                open_markup_stack.pop(i)
                                break
                            # If we're inside a token, we need to handle crossing XML
                            # The element should close, and the token will continue outside
                            if current_tok_elem is not None:
                                # Check if token is a descendant of the element to close
                                check_elem = current_tok_elem
                                token_inside_element = False
                                while check_elem is not None:
                                    if check_elem == open_elem:
                                        token_inside_element = True
                                        break
                                    check_elem = parent_map.get(check_elem)
                                
                                if token_inside_element:
                                    # Token is inside the element and spans across its boundary
                                    # The element ends, but the token continues outside it
                                    tok_start, tok_end, token = token_positions[token_idx] if token_idx < len(token_positions) else (None, None, None)
                                    parent_of_element = parent_map.get(open_elem, block_elem)
                                    merged_rejoin = (
                                        merged_form_for_display_span(
                                            tok_start, tok_end, rejoin_spans
                                        )
                                        if rejoin_spans
                                        and tok_start is not None
                                        and tok_end is not None
                                        else None
                                    )
                                    if merged_rejoin:
                                        # Line-break rejoin: preserve inline XML; do not splice
                                        # plaintext[char_pos:tok_end] (drops hyphen/lb structure).
                                        markup_token_state.pop(open_elem, None)
                                        current_elem = parent_of_element
                                        open_markup_stack.pop(i)
                                        break

                                    if tok_end is not None and tok_end > char_pos:
                                        # Move token to be a sibling of the element (outside)
                                        token_parent = parent_map.get(current_tok_elem)
                                        if token_parent == open_elem:
                                            open_elem.remove(current_tok_elem)
                                            parent_children = list(parent_of_element)
                                            elem_index = parent_children.index(open_elem) if open_elem in parent_children else len(parent_children) - 1
                                            parent_of_element.insert(elem_index + 1, current_tok_elem)
                                            parent_map[current_tok_elem] = parent_of_element  # keep parent_map correct
                                        
                                        # Determine portion of the token captured while this markup was open
                                        state = markup_token_state.get(open_elem)
                                        text_len_before = state["text_len"] if state and state.get("token") == current_tok_elem else len(current_tok_elem.text or "")
                                        children_count_before = state["children_count"] if state and state.get("token") == current_tok_elem else len(current_tok_elem)
                                        
                                        token_text = current_tok_elem.text or ""
                                        moved_text = token_text[text_len_before:]
                                        remaining_text = token_text[:text_len_before]
                                        current_tok_elem.text = remaining_text if remaining_text else None
                                        
                                        moved_children = list(current_tok_elem)[children_count_before:]
                                        for child in moved_children:
                                            current_tok_elem.remove(child)
                                        
                                        # When token spans the boundary, ensure the reopened element gets
                                        # the text from plaintext[char_pos:tok_end] and skip adding those
                                        # chars again in the character loop.
                                        if tok_end is not None and tok_end > char_pos:
                                            suffix = plaintext[char_pos:tok_end]
                                            if suffix:
                                                moved_text = (moved_text or "") + suffix
                                                skip_char_until = tok_end
                                        
                                        if moved_text or moved_children:
                                            reopened_elem = ET.Element(open_elem.tag, dict(open_elem.attrib))
                                            reopened_elem.set("rpt", "1")
                                            if moved_text:
                                                reopened_elem.text = moved_text
                                            for child in moved_children:
                                                reopened_elem.append(child)
                                                parent_map[child] = reopened_elem
                                            current_tok_elem.insert(children_count_before, reopened_elem)
                                            parent_map[reopened_elem] = current_tok_elem
                                        
                                        # Further characters should go to the token, not the reopened element
                                        current_elem = current_tok_elem
                                        
                                        markup_token_state.pop(open_elem, None)
                                        current_elem = parent_of_element
                                        open_markup_stack.pop(i)
                                    else:
                                        # Token doesn't continue - just close
                                        markup_token_state.pop(open_elem, None)
                                        current_elem = parent_of_element
                                        open_markup_stack.pop(i)
                                    break
                                else:
                                    # Token is not inside the element - safe to close
                                    markup_token_state.pop(open_elem, None)
                                    current_elem = parent_map.get(open_elem, block_elem)
                                    open_markup_stack.pop(i)
                                    break
                            else:
                                # Not inside a token - safe to close
                                markup_token_state.pop(open_elem, None)
                                current_elem = parent_map.get(open_elem, block_elem)
                                open_markup_stack.pop(i)
                                break
        
        # Open elements that start at this position
        if char_pos in opens_at:
            # Sort to ensure correct nesting:
            # 1. Sentences and tokens: longest first (so sentence opens before token at same position)
            # 2. Markup: document order first so sibling elements (e.g. <hi><lb/></hi><hi>V</hi>)
            #    open in original order, not by length (which would put outer/longer first and reverse order)
            def get_open_key(item):
                item_type, item_data = item
                if item_type == 'sentence':
                    start_pos, end_pos = item_data[0], item_data[1]
                    length = end_pos - start_pos
                    # Tier 0: sentences (outermost)
                    return (0, -length, 0, -end_pos)
                elif item_type == 'token':
                    start_pos, end_pos = item_data[0], item_data[1]
                    length = end_pos - start_pos
                    # Tier 2: tokens after zero-width milestones at the same offset
                    return (2, -length, 0, -end_pos)
                else:  # markup
                    end_pos = item_data['end']
                    start_pos = item_data['start']
                    length = end_pos - start_pos
                    level = item_data.get('level', 0)
                    order = item_data.get('order', 0)
                    if start_pos == end_pos:
                        # Tier 1: <lb/>, <cb/>, etc. before <tok> at the same char index
                        return (1, order, level, -end_pos)
                    # Tier 3: regular markup in document order
                    return (3, order, -length, -end_pos, level)
            
            to_open = sorted(opens_at[char_pos], key=get_open_key)
            
            for open_type, open_data in to_open:
                if open_type == 'sentence':
                    sent_start, sent_end, sent = open_data
                    if sent_start == char_pos:
                        # Ensure previous sentence is closed before opening a new one
                        if current_sent_elem is not None:
                            # Previous sentence should have been closed - this shouldn't happen
                            # but if it does, close it first
                            parent_after_sentence = parent_map.get(current_sent_elem, block_elem)
                            if open_markup_stack:
                                current_elem = open_markup_stack[-1][1]
                            else:
                                current_elem = parent_after_sentence
                            current_sent_elem = None
                            sent_token_ids = []
                            sent_idx += 1
                        
                        # Open sentence
                        current_sentence_idx = sent_idx
                        current_sentence_is_fallback = current_sentence_idx in sentence_fallbacks
                        current_sentence_first_token = None
                        current_sentence_obj = sent
                        sent_token_ids = []
                        
                        if current_sentence_is_fallback:
                            current_sent_elem = None
                            continue
                        
                        s_elem = ET.Element("s")
                        s_id = f"s-{sent_idx + 1}"  # sent_idx already starts at start_sent_idx
                        s_elem.set("id", s_id)
                        if sent.text:
                            s_elem.set("text", sent.text)
                        current_elem.append(s_elem)
                        parent_map[s_elem] = current_elem
                        current_elem = s_elem
                        current_sent_elem = s_elem
                elif open_type == 'token':
                    tok_start, tok_end, token = open_data
                    if tok_start == char_pos:
                        # Skip tokens that are subtokens of MWT tokens (they're handled as part of the MWT)
                        if token.id and token.id in mwt_subtoken_ids:
                            continue
                        # Check if this is an MWT token with subtokens.
                        # An MWT may only use the form-as-text shortcut when its
                        # display span is a contiguous run of plain text. If the
                        # span straddles inline markup (a line-broken / hyphenated
                        # word) or was clamped at a milestone, render it as a plain
                        # token so the surface is rebuilt from the original text
                        # and markup instead of duplicating the combined form.
                        if (
                            token.is_mwt
                            and token.subtokens
                            and _mwt_surface_is_plain(
                                tok_start, tok_end, token, plaintext, markup, block_elem
                            )
                        ):
                            # Create parent <tok> element with combined form
                            tok_elem = ET.Element("tok")
                            tok_id = f"w-{global_tok_id}"
                            global_tok_id += 1
                            tok_elem.set("id", tok_id)
                            # Set the combined form as text content
                            tok_elem.text = token.form
                            
                            # Store mapping from ord to tokid for head conversion (use first subtoken's id)
                            if token.subtokens and token.subtokens[0].id:
                                ord_to_tokid[token.subtokens[0].id] = tok_id
                            
                            # Create <dtok> elements for each subtoken
                            for dtok_idx, subtoken in enumerate(token.subtokens):
                                dtok_elem = ET.Element("dtok")
                                dtok_id = f"{tok_id}.{dtok_idx + 1}"
                                dtok_elem.set("id", dtok_id)
                                dtok_elem.set("form", subtoken.form)
                                
                                # Store mapping from subtoken ord to dtok id for head conversion
                                if subtoken.id:
                                    ord_to_tokid[subtoken.id] = dtok_id
                                
                                # Set all annotations on <dtok>
                                # Check both direct fields and attrs dict (backends may store in either)
                                dtok_lemma = subtoken.lemma or subtoken.attrs.get("lemma", "")
                                dtok_xpos = subtoken.xpos or subtoken.attrs.get("xpos", "")
                                dtok_upos = subtoken.upos or subtoken.attrs.get("upos", "")
                                dtok_feats = subtoken.feats or subtoken.attrs.get("feats", "")
                                
                                _set_attr(dtok_elem, "lemma", dtok_lemma, default_empty=True)
                                _set_attr(dtok_elem, "xpos", dtok_xpos)
                                _set_attr(dtok_elem, "upos", dtok_upos)
                                _set_attr(dtok_elem, "feats", dtok_feats)
                                
                                # Set other attributes
                                _set_attr(dtok_elem, "reg", subtoken.reg or "")
                                _set_attr(dtok_elem, "expan", subtoken.expan or "")
                                _set_attr(dtok_elem, "mod", subtoken.mod or "")
                                _set_attr(dtok_elem, "trslit", subtoken.trslit or "")
                                _set_attr(dtok_elem, "ltrslit", subtoken.ltrslit or "")
                                
                                # Set ord attribute
                                if subtoken.id:
                                    _set_attr(dtok_elem, "ord", str(subtoken.id))
                                
                                # Set head attribute (convert from ord to tokid)
                                # SubToken doesn't have head directly, check attrs
                                dtok_head_int = 0
                                if "head" in subtoken.attrs:
                                    head_attr = subtoken.attrs["head"]
                                    if isinstance(head_attr, int):
                                        dtok_head_int = head_attr
                                    elif isinstance(head_attr, str):
                                        try:
                                            dtok_head_int = int(head_attr)
                                        except (ValueError, TypeError):
                                            dtok_head_int = 0
                                
                                if dtok_head_int > 0:
                                    head_tokid = ord_to_tokid.get(dtok_head_int)
                                    if head_tokid:
                                        _set_attr(dtok_elem, "head", head_tokid)
                                    else:
                                        # Head not found yet, use ord for now
                                        _set_attr(dtok_elem, "head", str(dtok_head_int))
                                else:
                                    _set_attr(dtok_elem, "head", "", default_empty=True)
                                
                                # Set deprel attribute (check attrs dict)
                                dtok_deprel = subtoken.attrs.get("deprel", "")
                                _set_attr(dtok_elem, "deprel", dtok_deprel, default_empty=True)
                                
                                # Set deps attribute (check attrs dict)
                                dtok_deps = subtoken.attrs.get("deps", "")
                                if dtok_deps:
                                    _set_attr(dtok_elem, "deps", dtok_deps)
                                
                                # Set misc attribute (check attrs dict)
                                dtok_misc = subtoken.attrs.get("misc", "")
                                if dtok_misc:
                                    _set_attr(dtok_elem, "misc", dtok_misc)
                                
                                tok_elem.append(dtok_elem)
                            
                            current_elem.append(tok_elem)
                            # Update parent map
                            parent_map[tok_elem] = current_elem
                            # Don't change current_elem - MWT tokens don't contain other elements
                            current_tok_elem = tok_elem
                        else:
                            # Regular (non-MWT) token
                            tok_elem = ET.Element("tok")
                            tok_id = f"w-{global_tok_id}"
                            global_tok_id += 1
                            tok_elem.set("id", tok_id)  # id is always "id", not mapped
                            merged = merged_form_for_display_span(
                                tok_start, tok_end, rejoin_spans
                            ) if rejoin_spans else None
                            form_val = merged if merged else token.form
                            tok_elem.set("form", form_val)
                            
                            # Store mapping from ord to tokid for head conversion
                            if token.id:
                                ord_to_tokid[token.id] = tok_id
                            
                            _set_attr(tok_elem, "lemma", token.lemma or "", default_empty=True)
                            _set_attr(tok_elem, "xpos", token.xpos or "")
                            _set_attr(tok_elem, "upos", token.upos or "")
                            _set_attr(tok_elem, "feats", token.feats or "")
                            
                            # Set ord attribute
                            if token.id:
                                _set_attr(tok_elem, "ord", str(token.id))
                            
                            # Set head attribute (convert from ord to tokid)
                            head_ord = token.head
                            if head_ord and head_ord > 0:
                                head_tokid = ord_to_tokid.get(head_ord)
                                if head_tokid:
                                    _set_attr(tok_elem, "head", head_tokid)
                                else:
                                    # Head not found yet (might be later in document), use ord for now
                                    _set_attr(tok_elem, "head", str(head_ord))
                            else:
                                _set_attr(tok_elem, "head", "", default_empty=True)
                            
                            # Set deprel attribute
                            _set_attr(tok_elem, "deprel", token.deprel or "", default_empty=True)
                            
                            # Set misc attribute
                            if token.misc:
                                _set_attr(tok_elem, "misc", token.misc)
                            
                            current_elem.append(tok_elem)
                            # Update parent map
                            parent_map[tok_elem] = current_elem
                            current_elem = tok_elem
                            current_tok_elem = tok_elem
                        
                        if current_sentence_first_token is None:
                            current_sentence_first_token = tok_elem
                        
                        if current_sent_elem is not None:
                            sent_token_ids.append(tok_id)
                        elif current_sentence_is_fallback:
                            sent_token_ids.append(tok_id)
                        # Record starting state for any open markup elements
                        for _, open_elem in open_markup_stack:
                            markup_token_state[open_elem] = {
                                "token": tok_elem,
                                "text_len": len(tok_elem.text or ""),
                                "children_count": len(tok_elem),
                            }
                elif open_type == 'markup':
                    m = open_data
                    # Skip if it's the block element itself
                    if m['element'] == block_elem:
                        continue
                    
                    # Check if it's an extract element (like note)
                    tag_name = m['name']
                    self_closing_tags = {'lb', 'pb', 'cb', 'milestone', 'anchor', 'gap', 'fw'}
                    is_self_closing = tag_name in self_closing_tags
                    
                    if tag_name in extract_elements:
                        # Insert extract element (empty, will be restored later)
                        new_elem = ET.Element(m['element'].tag, m['element'].attrib)
                        current_elem.append(new_elem)
                        # Update parent map
                        parent_map[new_elem] = current_elem
                        # Don't change current_elem for extract elements
                    elif is_self_closing:
                        # Self-closing element: insert as empty, handle tail separately
                        new_elem = ET.Element(m['element'].tag, m['element'].attrib)
                        insert_parent = current_elem
                        # Milestone inside an already-open token span (e.g. rejoin merged
                        # "wšewědau<pc>-</pc><lb/>cností" into one UDPipe token) belongs
                        # inside <tok> after markup already emitted, not before the token.
                        if (
                            current_tok_elem is not None
                            and token_idx < len(token_positions)
                        ):
                            ts, te, _ = token_positions[token_idx]
                            if ts < m['start'] < te:
                                insert_parent = current_tok_elem
                        insert_parent.append(new_elem)
                        parent_map[new_elem] = insert_parent
                        self_closing_elements[(m['start'], m['end'])] = new_elem
                        # Don't add to stack - self-closing elements don't need to be closed
                        # Don't change current_elem - we want to continue adding to the parent
                    else:
                        # Regular element
                        new_elem = ET.Element(m['element'].tag, m['element'].attrib)
                        if m.get("split_reopen"):
                            new_elem.set("rpt", "1")
                        current_elem.append(new_elem)
                        # Update parent map
                        parent_map[new_elem] = current_elem
                        current_elem = new_elem
                        open_markup_stack.append((m, new_elem))
                        if current_tok_elem is not None:
                            markup_token_state[new_elem] = {
                                "token": current_tok_elem,
                                "text_len": len(current_tok_elem.text or ""),
                                "children_count": len(current_tok_elem),
                            }
        
        # Add character to current element (if we're inside a token)
        if char is not None:
            # Skip characters that were already added via a reopened element (markup boundary + token span)
            if skip_char_until is not None:
                if char_pos < skip_char_until:
                    continue
                skip_char_until = None
            # Check if this character belongs to a self-closing element's tail
            # This is character-by-character routing based on position
            # First, check if we're inside a token and if char_pos is in any tail range
            if token_idx < len(token_positions):
                tok_start, tok_end, token = token_positions[token_idx]
                if tok_start <= char_pos < tok_end and current_tok_elem is not None:
                    # Check if char_pos is in the tail range of any self-closing element
                    # that's a child of current_tok_elem
                    tail_elem_info = char_to_tail_elem.get(char_pos)
                    if tail_elem_info:
                        m, tail_start, tail_end = tail_elem_info
                        # Check if char_pos is in the tail range
                        if tail_start <= char_pos < tail_end:
                            # Find the corresponding element
                            tail_elem = None
                            # First try to find it in self_closing_elements
                            for (start, end), elem in self_closing_elements.items():
                                if start == m['start'] and end == m['end'] and elem.tag == m['name']:
                                    # Verify it's inside current_tok_elem
                                    if parent_map.get(elem) == current_tok_elem:
                                        tail_elem = elem
                                        break
                            # If not found, look in current_tok_elem's children
                            if tail_elem is None:
                                for child in current_tok_elem:
                                    if child.tag == m['name']:
                                        # Check attributes match (if any)
                                        if not m.get('contents') or all(child.get(k) == v for k, v in m.get('contents', {}).items()):
                                            tail_elem = child
                                            # Store it in self_closing_elements for future lookups
                                            self_closing_elements[(m['start'], m['end'])] = child
                                            break
                            
                            # If we found the element, route the character to its tail
                            if tail_elem is not None:
                                if tail_elem.tail:
                                    tail_elem.tail += char
                                else:
                                    tail_elem.tail = char
                                # Skip the rest - we've handled this character
                                continue
            
            if token_idx < len(token_positions):
                tok_start, tok_end, token = token_positions[token_idx]
                if tok_start <= char_pos < tok_end:
                    # Inside token - add to token element
                    if current_tok_elem is not None:
                        check_elem = current_elem
                        is_inside_tok = False
                        while check_elem is not None:
                            if check_elem == current_tok_elem:
                                is_inside_tok = True
                                break
                            check_elem = parent_map.get(check_elem)
                        
                        if not is_inside_tok:
                            current_elem = current_tok_elem
                            is_inside_tok = True
                        
                        if current_elem == current_tok_elem:
                            # Check if this is an MWT token (has <dtok> children)
                            # MWT tokens already have their text set, so don't add characters
                            has_dtoks = any(c.tag.endswith("}dtok") or c.tag == "dtok" for c in current_tok_elem)
                            if not has_dtoks:
                                # Not an MWT token - append text as usual
                                if len(current_tok_elem):
                                    last_child = current_tok_elem[-1]
                                    last_tag = (
                                        last_child.tag.split("}")[-1]
                                        if isinstance(last_child.tag, str)
                                        else last_child.tag
                                    )
                                    sc_tags = {
                                        "lb", "pb", "cb",
                                        "milestone", "anchor", "gap", "fw",
                                    }
                                    tail_info = char_to_tail_elem.get(char_pos)
                                    if tail_info and tail_info[0].get("name") == last_tag:
                                        _, t_start, t_end = tail_info
                                        if t_start <= char_pos < t_end:
                                            if last_child.tail:
                                                last_child.tail += char
                                            else:
                                                last_child.tail = char
                                            continue
                                    if last_tag in sc_tags:
                                        if last_child.tail:
                                            last_child.tail += char
                                        else:
                                            last_child.tail = char
                                    elif last_child.tail:
                                        last_child.tail += char
                                    else:
                                        last_child.tail = char
                                else:
                                    if current_tok_elem.text:
                                        current_tok_elem.text += char
                                    else:
                                        current_tok_elem.text = char
                            # If it's an MWT token, skip adding text (already set when token was created)
                        else:
                            if not current_elem.text:
                                current_elem.text = char
                            else:
                                current_elem.text += char
                    else:
                        # Fallback
                        if tail_elem is not None:
                            if tail_elem.tail:
                                tail_elem.tail += char
                            else:
                                tail_elem.tail = char
                        else:
                            if not current_elem.text:
                                current_elem.text = char
                            else:
                                current_elem.text += char
                elif char_pos == tok_end:
                    # Close token (closes_at may have done this already)
                    if current_tok_elem is not None:
                        current_elem = parent_map.get(current_tok_elem, block_elem)
                        current_tok_elem = None
                    token_idx += 1
                    # Do not emit bare text when the next token starts at this offset
                    # (e.g. "našemu" + "!" with SpaceAfter=No).
                    next_starts_here = (
                        token_idx < len(token_positions)
                        and token_positions[token_idx][0] == char_pos
                    )
                    if next_starts_here:
                        continue
                    # Add character after token
                    if char_pos < len(plaintext):
                        if tail_elem is not None:
                            if tail_elem.tail:
                                tail_elem.tail += char
                            else:
                                tail_elem.tail = char
                        else:
                            if not current_elem.text and not list(current_elem):
                                current_elem.text = char
                            elif list(current_elem):
                                if current_elem[-1].tail:
                                    current_elem[-1].tail += char
                                else:
                                    current_elem[-1].tail = char
                            else:
                                if current_elem.text:
                                    current_elem.text += char
                                else:
                                    current_elem.text = char
                else:
                    # Before token - add to current element
                    if tail_elem is not None:
                        if tail_elem.tail:
                            tail_elem.tail += char
                        else:
                            tail_elem.tail = char
                    else:
                        # Normal placement
                        if not current_elem.text and not list(current_elem):
                            current_elem.text = char
                        elif list(current_elem):
                            if current_elem[-1].tail:
                                current_elem[-1].tail += char
                            else:
                                current_elem[-1].tail = char
                        else:
                            if current_elem.text:
                                current_elem.text += char
                            else:
                                current_elem.text = char
            else:
                # No more tokens - add to current element
                if tail_elem is not None:
                    if tail_elem.tail:
                        tail_elem.tail += char
                    else:
                        tail_elem.tail = char
                else:
                    # Normal placement
                    if not current_elem.text and not list(current_elem):
                        current_elem.text = char
                    elif list(current_elem):
                        if current_elem[-1].tail:
                            current_elem[-1].tail += char
                        else:
                            current_elem[-1].tail = char
                    else:
                        if current_elem.text:
                            current_elem.text += char
                        else:
                            current_elem.text = char
        
        # Close sentence if needed
        if sent_idx < len(sentence_positions):
            sent_start, sent_end, sent = sentence_positions[sent_idx]
            if char_pos == sent_end:
                # Check if sentence is empty
                if current_sent_elem is not None:
                    has_text = current_sent_elem.text and current_sent_elem.text.strip()
                    has_children = len(list(current_sent_elem)) > 0
                    if not has_text and not has_children and sent_token_ids:
                        current_sent_elem.set("corresp", " ".join(f"#{tid}" for tid in sent_token_ids))
                    
                    # Close sentence
                    current_elem = parent_map.get(current_sent_elem, block_elem)
                    current_sent_elem = None
                elif current_sentence_is_fallback and sent_token_ids:
                    s_elem = ET.Element("s")
                    s_id = f"s-{sent_idx + 1}"  # sent_idx already starts at start_sent_idx
                    
                    s_elem.set("id", s_id)
                    if current_sentence_obj and current_sentence_obj.text:
                        s_elem.set("text", current_sentence_obj.text)
                    s_elem.set("corresp", " ".join(f"#{tid}" for tid in sent_token_ids))
                    
                    target_parent = parent_map.get(current_sentence_first_token, block_elem)
                    if target_parent is not None:
                        children = list(target_parent)
                        try:
                            insert_idx = children.index(current_sentence_first_token)
                        except ValueError:
                            insert_idx = 0
                        target_parent.insert(insert_idx, s_elem)
                    else:
                        block_elem.append(s_elem)
                        target_parent = block_elem
                    parent_map[s_elem] = target_parent
                
                sent_token_ids = []
                current_sentence_idx = None
                current_sentence_is_fallback = False
                current_sentence_first_token = None
                current_sentence_obj = None
                sent_idx += 1
    
    # Restore original tail whitespace for this block element
    if original_tail is not None:
        block_elem.tail = original_tail
    
    # Second pass: Update all head values from ord to tokid
    # Now that all tokens are created, we can convert head ord values to tokids
    for tok_elem in block_elem.iter():
        if tok_elem.tag.endswith("}tok") or tok_elem.tag == "tok":
            # Get current head value
            head_attr = _resolve_attr_name("head")
            if head_attr:
                head_value = tok_elem.get(head_attr)
                if head_value and head_value.isdigit():
                    # It's an ord value, try to convert to tokid
                    head_ord = int(head_value)
                    head_tokid = ord_to_tokid.get(head_ord)
                    if head_tokid:
                        _set_attr(tok_elem, "head", head_tokid)
                    elif head_ord == 0:
                        # Root token, remove head attribute
                        _set_attr(tok_elem, "head", "", default_empty=True)
    
    # Return number of tokens and sentences used
    tokens_used = global_tok_id - initial_tok_id
    sentences_used = sent_idx - initial_sent_idx
    return (tokens_used, sentences_used)


def _plaintext_to_xml_positions(xml_str: str, plaintext: str) -> List[Tuple[int, int, int, int]]:
    """
    Map plaintext character ranges to positions in the serialized XML string.
    Single pass: skip tags (from < to >, respecting quoted attrs). Each run of
    text in XML maps to a contiguous span of plaintext. Spacing is preserved.
    Returns list of (plaintext_start, plaintext_end, xml_start, xml_end).
    """
    spans: List[Tuple[int, int, int, int]] = []
    xml_pos = 0
    plaintext_pos = 0
    n_plain = len(plaintext)
    n_xml = len(xml_str)
    in_tag = False
    run_plain_start = 0
    run_xml_start = 0
    while xml_pos < n_xml:
        c = xml_str[xml_pos]
        if c == "<":
            if not in_tag and plaintext_pos > run_plain_start:
                spans.append((run_plain_start, plaintext_pos, run_xml_start, xml_pos))
            in_tag = True
            xml_pos += 1
            continue
        if in_tag:
            if c in ("\"", "'"):
                quote = c
                xml_pos += 1
                while xml_pos < n_xml and xml_str[xml_pos] != quote:
                    if xml_str[xml_pos] == "\\":
                        xml_pos += 1
                    xml_pos += 1
                if xml_pos < n_xml:
                    xml_pos += 1
                continue
            if c == ">":
                in_tag = False
                run_plain_start = plaintext_pos
                run_xml_start = xml_pos + 1
            xml_pos += 1
            continue
        if plaintext_pos < n_plain:
            plaintext_pos += 1
        xml_pos += 1
    if not in_tag and run_xml_start < n_xml and plaintext_pos > run_plain_start:
        spans.append((run_plain_start, plaintext_pos, run_xml_start, xml_pos))
    return spans


def _plaintext_pos_to_xml(
    spans: List[Tuple[int, int, int, int]],
    plaintext_pos: int,
    after: bool,
) -> int:
    """XML string index to insert at: before plaintext_pos (tag before char at P) or after (tag after char at P-1)."""
    for p_start, p_end, x_start, x_end in spans:
        if after:
            if p_start < plaintext_pos <= p_end:
                return x_start + (plaintext_pos - p_start)
        else:
            if p_start <= plaintext_pos < p_end:
                return x_start + (plaintext_pos - p_start)
            if plaintext_pos == p_start:
                return x_start
    if not spans:
        return 0
    if after and plaintext_pos >= spans[-1][1]:
        return spans[-1][3]
    return spans[0][2]


def rebuild_xml_with_tokens_string_based(
    block_elem: ET.Element,
    plaintext: str,
    markup: List[Dict[str, Any]],
    token_positions: List[Tuple[int, int, Token]],
    sentence_positions: List[Tuple[int, int, Sentence]],
    document: Document,
    block_elements: Set[str],
    extract_elements: Set[str],
    note_content_map: Dict[str, Tuple[str, Dict[str, str], List[ET.Element]]],
    start_tok_id: int = 1,
    start_sent_idx: int = 0,
    settings: Optional[Any] = None,
) -> Tuple[int, int]:
    """
    Rebuild block by working on the serialized XML string. Serialization already
    has correct spacing (text + tails). We map plaintext offsets to positions
    in that string with a simple scan (no regex), then insert <s>/<tok> at
    those positions and re-parse. Avoids all .text/.tail mutation.

    Limitation: when a sentence spans across existing tag boundaries (e.g.
    </hi><hi>), inserting </s> at the last character can wrap a literal "</hi>"
    inside <s>, producing invalid XML. This engine is safest for blocks with
    little or no nested markup; for complex nesting use standoff or minidom.
    """
    def _resolve_attr(internal_attr: str) -> str:
        if settings:
            return settings.resolve_xml_attribute(internal_attr, default=internal_attr) or internal_attr
        return internal_attr

    xml_str = ET.tostring(block_elem, encoding="unicode")
    spans = _plaintext_to_xml_positions(xml_str, plaintext)
    if not spans:
        return (0, 0)

    insertions: List[Tuple[int, str, int]] = []
    global_tok_id = start_tok_id
    sent_idx = start_sent_idx
    id_attr = _resolve_attr("id")

    for s_start, s_end, sent in sentence_positions:
        pos_before = _plaintext_pos_to_xml(spans, s_start, after=False)
        pos_after = _plaintext_pos_to_xml(spans, s_end, after=True)
        s_id = f"s-{sent_idx + 1}"
        attrs = f' {id_attr}="{s_id}"'
        if getattr(sent, "text", None):
            from xml.sax.saxutils import escape
            attrs += f' text="{escape(sent.text)}"'
        insertions.append((pos_after, "</s>", 0))
        insertions.append((pos_before, f"<s{attrs}>", 0))
        sent_idx += 1

    for t_start, t_end, token in token_positions:
        pos_before = _plaintext_pos_to_xml(spans, t_start, after=False)
        pos_after = _plaintext_pos_to_xml(spans, t_end, after=True)
        tok_id = f"w-{global_tok_id}"
        insertions.append((pos_after, "</tok>", 0))
        insertions.append((pos_before, f"<tok {id_attr}=\"{tok_id}\">", 1))
        global_tok_id += 1

    # Sort by position descending (insert from end so indices stay valid), then by order so at same position we get <tok> before <s> (so result is <s><tok>...)
    insertions.sort(key=lambda x: (-x[0], -x[2]))
    for pos, tag, _ in insertions:
        xml_str = xml_str[:pos] + tag + xml_str[pos:]

    new_elem = ET.fromstring(xml_str)
    original_tail = block_elem.tail
    block_elem.clear()
    block_elem.attrib.update(dict(new_elem.attrib))
    for child in list(new_elem):
        block_elem.append(child)
    if original_tail is not None:
        block_elem.tail = original_tail

    return (global_tok_id - start_tok_id, sent_idx - start_sent_idx)

