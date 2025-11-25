from __future__ import annotations

import re
import string
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .doc import Document, Sentence, SubToken, Token, Entity, UD_DOCUMENT_ATTRIBUTES, UD_SENTENCE_ATTRIBUTES
from .doc_utils import collect_span_entities_by_sentence

DEFAULT_GENERATOR = "flexipipe"


def _is_punctuation(token: Token) -> bool:
    """Check if a token is punctuation based on UPOS or form."""
    # Check UPOS first (most reliable)
    if token.upos == "PUNCT":
        return True
    # Fallback: check if form is a single punctuation character
    if len(token.form) == 1 and token.form in string.punctuation:
        return True
    return False


def _create_implicit_mwt(sentence: Sentence) -> Sentence:
    """
    Create implicit MWTs from sequences of tokens with SpaceAfter=No (excluding punctuation).
    
    This converts sequences like:
    29  set  ...  SpaceAfter=No
    30  -    ...  SpaceAfter=No  (punctuation - skip)
    31  up   ...  SpaceAfter=No
    
    Into:
    29-31  set-up  _  _  _  _  _  _  _  _
    29     set     ... (as subtoken)
    30     -       ... (as subtoken, but punctuation is included)
    31     up      ... (as subtoken)
    
    Only creates MWTs if no punctuation is involved in the sequence.
    """
    if not sentence.tokens:
        return sentence
    
    new_sentence = Sentence(
        id=sentence.id,
        sent_id=sentence.sent_id,
        text=sentence.text,
        tokens=[],
        entities=list(sentence.entities),  # Preserve entities
        attrs=dict(sentence.attrs),  # Preserve attributes
        source_id=sentence.source_id,
        char_start=sentence.char_start,
        char_end=sentence.char_end,
        byte_start=sentence.byte_start,
        byte_end=sentence.byte_end,
        corr=sentence.corr,
    )
    
    i = 0
    while i < len(sentence.tokens):
        token = sentence.tokens[i]
        
        # Skip if already an MWT
        if token.is_mwt:
            new_sentence.tokens.append(token)
            i += 1
            continue
        
        # Look for a sequence of tokens with SpaceAfter=No
        sequence = [token]
        j = i + 1
        
        while j < len(sentence.tokens):
            next_token = sentence.tokens[j]
            # Stop if previous token doesn't have SpaceAfter=No
            if sequence[-1].space_after is not False:
                break
            # Stop if next token is already an MWT
            if next_token.is_mwt:
                break
            # Include the next token
            sequence.append(next_token)
            j += 1
        
        # Only create MWT if we have at least 2 tokens
        if len(sequence) >= 2:
            # Check if any token in sequence is punctuation
            # But allow apostrophes/single quotes in contractions (e.g., "Let's", "don't")
            has_punctuation = any(_is_punctuation(tok) for tok in sequence)
            # Allow apostrophes in contractions: if punctuation is only apostrophe-like and at position 1
            # (e.g., "Let" + "'s" or "do" + "n't")
            is_contraction = False
            if has_punctuation and len(sequence) == 2:
                punct_token = sequence[1] if _is_punctuation(sequence[1]) else None
                if punct_token:
                    # Check if it's an apostrophe-like token (starts with ' or contains apostrophe)
                    form = punct_token.form
                    if form.startswith("'") or form.startswith("'") or "'" in form:
                        is_contraction = True
            
            if not has_punctuation or is_contraction:
                # Create MWT
                parent_form = "".join(tok.form for tok in sequence)
                parent = Token(
                    id=sequence[0].id,
                    form=parent_form,
                    lemma="_",
                    xpos="_",
                    upos="_",
                    feats="_",
                    is_mwt=True,
                    mwt_start=sequence[0].id,
                    mwt_end=sequence[-1].id,
                    space_after=sequence[-1].space_after,  # Inherit from last token
                    subtokens=[],
                )
                
                # Add subtokens
                for tok in sequence:
                    subtoken = SubToken(
                        id=tok.id,
                        form=tok.form,
                        lemma=tok.lemma,
                        upos=tok.upos,
                        xpos=tok.xpos,
                        feats=tok.feats,
                        reg=tok.reg,
                        expan=tok.expan,
                        mod=tok.mod,
                        trslit=tok.trslit,
                        ltrslit=tok.ltrslit,
                        tokid=tok.tokid,
                        space_after=False,  # No space after subtokens (they're not orthographic)
                    )
                    parent.subtokens.append(subtoken)
                
                new_sentence.tokens.append(parent)
                i = j
            else:
                # Has punctuation - don't create MWT, just add tokens normally
                new_sentence.tokens.append(token)
                i += 1
        else:
            # Single token or sequence doesn't qualify - add normally
            new_sentence.tokens.append(token)
            i += 1
    
    return new_sentence


def _merge_spaceafter_no_contractions(document: Document) -> None:
    """
    Merge adjacent non-punctuation tokens with SpaceAfter=No into contractions.
    
    Some treebanks represent contractions as two separate tokens without a space
    between them (indicated by SpaceAfter=No). This function converts these into
    contractions (MWTs) for internal processing, ensuring tokenization matches.
    """
    for sentence in document.sentences:
        if not sentence.tokens:
            continue
        
        new_tokens: List[Token] = []
        i = 0
        
        while i < len(sentence.tokens):
            current = sentence.tokens[i]
            
            # Check if we should merge with next token
            # Skip if current token is already part of an MWT (has subtokens)
            if (i + 1 < len(sentence.tokens) and 
                not current.subtokens and  # Not already an MWT
                current.space_after is False and  # SpaceAfter=No
                not _is_punctuation(current) and  # Current is not punctuation
                not _is_punctuation(sentence.tokens[i + 1]) and  # Next is not punctuation
                not sentence.tokens[i + 1].subtokens):  # Next is not already an MWT
                
                # Merge current and next token into a contraction
                next_token = sentence.tokens[i + 1]
                
                # Create parent token (contraction)
                parent_form = current.form + next_token.form
                parent = Token(
                    id=current.id,
                    form=parent_form,
                    lemma=current.lemma,
                    upos=current.upos,  # Use first token's UPOS
                    xpos=current.xpos,  # Use first token's XPOS
                    feats=current.feats,
                    reg=current.reg,
                    mod=current.mod,
                    expan=current.expan,
                    trslit=current.trslit,
                    ltrslit=current.ltrslit,
                    tokid=current.tokid,
                    is_mwt=True,
                    mwt_start=current.id,
                    mwt_end=next_token.id,
                    space_after=next_token.space_after,  # Inherit space_after from second token
                    source=current.source,
                    head=current.head,
                    deprel=current.deprel,
                    deps=current.deps,
                    misc=current.misc,
                )
                
                # Create subtokens
                subtoken1 = SubToken(
                    id=current.id,
                    form=current.form,
                    lemma=current.lemma,
                    upos=current.upos,
                    xpos=current.xpos,
                    feats=current.feats,
                    reg=current.reg,
                    mod=current.mod,
                    expan=current.expan,
                    trslit=current.trslit,
                    ltrslit=current.ltrslit,
                    tokid=current.tokid,
                    source=current.source,
                    space_after=False,  # No space after first part
                )
                
                subtoken2 = SubToken(
                    id=next_token.id,
                    form=next_token.form,
                    lemma=next_token.lemma,
                    upos=next_token.upos,
                    xpos=next_token.xpos,
                    feats=next_token.feats,
                    reg=next_token.reg,
                    mod=next_token.mod,
                    expan=next_token.expan,
                    trslit=next_token.trslit,
                    ltrslit=next_token.ltrslit,
                    tokid=next_token.tokid,
                    source=next_token.source,
                    space_after=next_token.space_after if next_token.space_after is not None else True,
                )
                
                parent.subtokens = [subtoken1, subtoken2]
                new_tokens.append(parent)
                i += 2  # Skip both tokens
            else:
                # No merge needed, keep token as-is
                new_tokens.append(current)
                i += 1
        
        sentence.tokens = new_tokens

def conllu_to_document(conllu_text: str, doc_id: str | None = None, add_tokids: bool = False) -> Document:
    """
    Parse CoNLL-U text into a Document.
    
    Args:
        conllu_text: CoNLL-U formatted text
        doc_id: Optional document ID
        add_tokids: If True, add unique token IDs (s1-t1 format) when TokId is missing in MISC
    """
    document = Document(id=doc_id or "")
    current_sentence: Sentence | None = None
    # Temporary storage for pending MWT ranges: start_id -> (end_id, form)
    pending_mwt: dict[int, tuple[int, str]] = {}
    # Map from token id to Token created for MWT range
    mwt_parents: dict[int, Token] = {}
    sentence_counter = 1  # Start at 1 for s1, s2, etc.
    seen_any_token = False
    for raw_line in conllu_text.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            if current_sentence:
                # Generate sentid if missing and add_tokids is True
                if add_tokids and not current_sentence.sent_id:
                    current_sentence.sent_id = f"s{sentence_counter}"
                    if not current_sentence.id:
                        current_sentence.id = current_sentence.sent_id
                document.sentences.append(current_sentence)
                current_sentence = None
                pending_mwt.clear()
                mwt_parents.clear()
                sentence_counter += 1
            continue
        if line.startswith("#"):
            key_val = line[1:].strip()
            if key_val.startswith("newdoc id"):
                parts = key_val.split("=", 1)
                if len(parts) == 2:
                    value = parts[1].strip()
                    document.id = value
                    document.set_standard_attr("newdoc_id", value)
                continue
            if key_val.startswith("newdoc"):
                document.attrs.setdefault("newdoc", key_val)
                continue
            if "=" in key_val:
                key, value = key_val.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Determine if this is a sentence-level attribute
                is_sentence_attr = key in UD_SENTENCE_ATTRIBUTES or key in {"sent_id", "text"}
                
                if is_sentence_attr:
                    # Sentence-level attribute - ensure sentence exists
                    if not current_sentence:
                        current_sentence = Sentence(id="", sent_id="")
                    if key == "sent_id":
                        current_sentence.set_standard_attr("sent_id", value)
                        if not current_sentence.id:
                            current_sentence.id = value
                        current_sentence.source_id = value
                    elif key == "text":
                        current_sentence.set_standard_attr("text", value)
                    elif key in UD_SENTENCE_ATTRIBUTES:
                        current_sentence.set_standard_attr(key, value)
                    else:
                        current_sentence.attrs[key] = value
                elif not current_sentence and not seen_any_token:
                    # Document-level attribute (before any sentence)
                    if key in UD_DOCUMENT_ATTRIBUTES:
                        document.set_standard_attr(key, value)
                    else:
                        document.attrs[key] = value
                elif current_sentence:
                    # Non-standard sentence-level attribute
                    current_sentence.attrs[key] = value
            if not current_sentence:
                current_sentence = Sentence(id="", sent_id="")
            continue
        # Token or MWT
        cols = line.split("\t")
        if len(cols) < 10:
            # Skip malformed lines
            continue
        seen_any_token = True
        tid = cols[0]
        form = cols[1]
        lemma = cols[2] if cols[2] != "_" else ""
        upos = cols[3] if cols[3] != "_" else ""
        xpos = cols[4] if cols[4] != "_" else ""
        feats = cols[5] if cols[5] != "_" else ""
        head_str = cols[6] if cols[6] != "_" else "0"
        deprel = cols[7] if cols[7] != "_" else ""
        deps = cols[8] if cols[8] != "_" else ""
        misc = cols[9] if cols[9] != "_" else ""
        
        # Parse head as integer
        try:
            head = int(head_str) if head_str else 0
        except (ValueError, TypeError):
            head = 0
        # Parse additional attributes from MISC
        reg = ""
        mod = ""
        expan = ""
        trslit = ""
        ltrslit = ""
        tokid = ""
        if misc:
            for part in misc.split("|"):
                if part.startswith("Normalization="):
                    reg = part[14:]  # len("Normalization=")
                elif part.startswith("ModernForm="):
                    mod = part[11:]  # len("ModernForm=")
                elif part.startswith("Expansion="):
                    expan = part[10:]  # len("Expansion=")
                elif part.startswith("Translit="):
                    trslit = part[9:]  # len("Translit=")
                elif part.startswith("LTransLit="):
                    ltrslit = part[11:]  # len("LTransLit=")
                elif part.startswith("TokId="):
                    tokid = part[6:]  # len("TokId=")
        
        # Generate tokid if missing and add_tokids is True
        if add_tokids and not tokid:
            # Use sentence counter and token id to create unique ID
            sent_id_str = str(sentence_counter) if current_sentence else "0"
            if "-" in tid:
                # For MWT ranges, use the start ID
                start_s, _ = tid.split("-", 1)
                try:
                    token_id_str = start_s
                except ValueError:
                    token_id_str = "0"
            else:
                token_id_str = tid
            tokid = f"s{sent_id_str}-t{token_id_str}"
        
        if "-" in tid:
            # MWT range line
            start_s, end_s = tid.split("-", 1)
            try:
                start_id = int(start_s)
                end_id = int(end_s)
            except ValueError:
                continue
            pending_mwt[start_id] = (end_id, form)
            # Create parent token now to preserve order
            parent = Token(
                id=start_id,
                form=form,
                is_mwt=True,
                mwt_start=start_id,
                mwt_end=end_id,
                tokid=tokid,
            )
            # SpaceAfter on range from MISC
            parent.space_after = not ("SpaceAfter=No" in misc)
            if current_sentence is None:
                current_sentence = Sentence(id="")
            current_sentence.tokens.append(parent)
            mwt_parents[start_id] = parent
            continue
        # Regular or subtoken
        try:
            token_id = int(tid)
        except ValueError:
            continue
        if current_sentence is None:
            current_sentence = Sentence(id="")
        # Determine if this token falls inside a pending MWT
        parent_for_sub: Token | None = None
        for start_id, (end_id, _) in list(pending_mwt.items()):
            if start_id <= token_id <= end_id:
                parent_for_sub = mwt_parents.get(start_id)
                if token_id == end_id:
                    # Last child closes this MWT
                    pending_mwt.pop(start_id, None)
                break
        if parent_for_sub:
            # Subtoken
            # Generate tokid for subtoken if missing and add_tokids is True
            sub_tokid = tokid
            if add_tokids and not sub_tokid:
                sent_id_str = str(sentence_counter) if current_sentence else "0"
                sub_tokid = f"s{sent_id_str}-t{token_id}"
            
            st = SubToken(
                id=token_id,
                form=form,
                lemma=lemma,
                upos=upos,
                xpos=xpos,
                feats=feats,
                reg=reg,
                mod=mod,
                expan=expan,
                trslit=trslit,
                ltrslit=ltrslit,
                tokid=sub_tokid,
                space_after=True,  # Default; SpaceAfter only matters on range in CoNLL-U
            )
            parent_for_sub.subtokens.append(st)
        else:
            tok = Token(
                id=token_id,
                form=form,
                lemma=lemma,
                upos=upos,
                xpos=xpos,
                feats=feats,
                reg=reg,
                mod=mod,
                expan=expan,
                trslit=trslit,
                ltrslit=ltrslit,
                tokid=tokid,
                head=head,
                deprel=deprel,
                deps=deps,
                misc=misc if misc not in {"", None} else "_",
                space_after=not ("SpaceAfter=No" in misc),
            )
            current_sentence.tokens.append(tok)
    # Flush any last sentence
    if current_sentence:
        # Generate sentid if missing and add_tokids is True
        if add_tokids and not current_sentence.sent_id:
            current_sentence.sent_id = f"s{sentence_counter}"
            if not current_sentence.id:
                current_sentence.id = current_sentence.sent_id
        document.sentences.append(current_sentence)
    
    # Merge adjacent non-punctuation tokens with SpaceAfter=No into contractions
    # This handles treebanks that represent contractions as two tokens without a space
    _merge_spaceafter_no_contractions(document)
    
    # Set space_after to None for the last token of each sentence (no SpaceAfter entry in CoNLL-U)
    # This must be done AFTER _merge_spaceafter_no_contractions, which may modify tokens
    for sent in document.sentences:
        if sent.tokens:
            sent.tokens[-1].space_after = None
    
    # If sentence text missing, reconstruct with simple heuristic
    for sent in document.sentences:
        if not getattr(sent, "text", ""):
            # Rebuild using our conllu writer to ensure consistency
            # but here, do a minimal reconstruction using space_after flags
            text_parts: list[str] = []
            for tok in sent.tokens:
                text_parts.append(tok.form)
                if tok.space_after:
                    text_parts.append(" ")
            if text_parts and text_parts[-1] == " ":
                text_parts.pop()
            sent.text = "".join(text_parts)
    return document

def document_to_conllu(
    document: Document,
    *,
    generator: str = DEFAULT_GENERATOR,
    model: Optional[str] = None,
    model_info: Optional[Dict[str, str]] = None,
    create_implicit_mwt: bool = False,
    entity_format: str = "iob",  # "iob" for Entity=B-PER format, "ne" for NE=ORG_3 format
    custom_misc_attrs: Optional[Dict[str, str]] = None,  # Map attr key -> MISC tag name (e.g., {"myattr": "MyTag"})
) -> str:
    lines: List[str] = []

    if generator:
        lines.append(f"# generator = {generator}")

    if model:
        lines.append(f"# model = {model}")

    if model_info:
        for key, value in model_info.items():
            lines.append(f"# {key} = {value}")

    # Print standard UD document attributes
    doc_standard_attrs = document.get_standard_attrs()
    if doc_standard_attrs:
        # newdoc_id is special - it's printed as "# newdoc id = ..."
        newdoc_id = doc_standard_attrs.pop("newdoc_id", None) or document.id or ""
        if newdoc_id:
            lines.append(f"# newdoc id = {newdoc_id}")
        else:
            lines.append("# newdoc")
        # Print other standard document attributes
        for attr_name, attr_desc in sorted(UD_DOCUMENT_ATTRIBUTES.items()):
            if attr_name != "newdoc_id" and attr_name in doc_standard_attrs:
                lines.append(f"# {attr_name} = {doc_standard_attrs[attr_name]}")
    else:
        doc_id = document.id or ""
        if doc_id:
            lines.append(f"# newdoc id = {doc_id}")
        else:
            lines.append("# newdoc")

    span_entities = collect_span_entities_by_sentence(document, "ner")

    for idx, sentence in enumerate(document.sentences):
        if lines:
            lines.append("")
        # Optionally create implicit MWTs from SpaceAfter=No sequences
        if create_implicit_mwt:
            sentence = _create_implicit_mwt(sentence)
        extra_entities = span_entities.get(idx)
        lines.extend(_sentence_lines(sentence, extra_entities=extra_entities, entity_format=entity_format, custom_misc_attrs=custom_misc_attrs))

    if not lines:
        return ""

    return "\n".join(lines) + "\n"


def _sentence_lines(sentence: Sentence, extra_entities: Optional[List[Entity]] = None, entity_format: str = "iob", custom_misc_attrs: Optional[Dict[str, str]] = None) -> List[str]:
    lines: List[str] = []
    # Print standard UD sentence attributes
    sent_standard_attrs = sentence.get_standard_attrs()
    if sent_standard_attrs:
        # sent_id and text are special - they're printed first
        sent_id = sent_standard_attrs.pop("sent_id", None) or (sentence.sent_id or getattr(sentence, "source_id", "") or "").strip()
        if sent_id:
            lines.append(f"# sent_id = {sent_id}")
        
        sentence_text = sent_standard_attrs.pop("text", None) or sentence.text or _reconstruct_sentence_text(sentence.tokens)
        if sentence_text:
            lines.append(f"# text = {sentence_text}")
        
        # Print other standard sentence attributes
        for attr_name in sorted(UD_SENTENCE_ATTRIBUTES.keys()):
            if attr_name not in ("sent_id", "text") and attr_name in sent_standard_attrs:
                lines.append(f"# {attr_name} = {sent_standard_attrs[attr_name]}")
    else:
        # Fallback to old behavior if no standard attrs
        sent_identifier = (sentence.sent_id or getattr(sentence, "source_id", "") or "").strip()
        if sent_identifier:
            lines.append(f"# sent_id = {sent_identifier}")

        sentence_text = sentence.text or _reconstruct_sentence_text(sentence.tokens)
        if sentence_text:
            lines.append(f"# text = {sentence_text}")

    # Build entity annotations for NER entities
    if entity_format == "ne":
        # NE= format: assign sequential IDs to entities of the same type
        # Map token ID (1-based) to NE=LABEL_ID
        token_ne_labels: dict[int, str] = {}  # token_id -> "ORG_3"
        label_counters: dict[str, int] = {}  # label -> next_id
        
        def _apply_entity_ne(entity: Entity) -> None:
            label = entity.label
            if label not in label_counters:
                label_counters[label] = 1
            entity_id = label_counters[label]
            label_counters[label] += 1
            
            ne_value = f"{label}_{entity_id}"
            for token_id in range(entity.start, entity.end + 1):
                if token_id <= 0:
                    continue
                if token_id > len(sentence.tokens):
                    break
                token_ne_labels[token_id] = ne_value
        
        if sentence.entities:
            for entity in sentence.entities:
                _apply_entity_ne(entity)
        if extra_entities:
            for entity in extra_entities:
                _apply_entity_ne(entity)
        
        # Pass NE labels to token formatting
        token_entity_labels = token_ne_labels
        use_ne_format = True
    else:
        # IOB format: Entity=B-PER, Entity=I-PER
        token_entity_labels: dict[int, str] = {}  # token_id -> "B-ORG" or "I-ORG"
        def _apply_entity_iob(entity: Entity) -> None:
            for token_id in range(entity.start, entity.end + 1):
                if token_id <= 0:
                    continue
                if token_id > len(sentence.tokens):
                    break
                if token_id == entity.start:
                    token_entity_labels[token_id] = f"B-{entity.label}"
                else:
                    token_entity_labels[token_id] = f"I-{entity.label}"

        if sentence.entities:
            for entity in sentence.entities:
                _apply_entity_iob(entity)
        if extra_entities:
            for entity in extra_entities:
                _apply_entity_iob(entity)
        
        use_ne_format = False

    current_id = 1
    pos = 0  # index into sentence_text for space derivation
    tokens_list = list(sentence.tokens)
    for token_idx, token in enumerate(tokens_list):
        is_last_token = (token_idx == len(tokens_list) - 1)
        if token.is_mwt and token.subtokens:
            # Derive space for the orthographic token (range) from sentence_text
            space_after_range: Optional[bool] = None
            # First, check if token has explicit space_after from input (e.g., from CoNLL-U)
            if token.space_after is not None:
                space_after_range = token.space_after
            elif sentence_text:
                idx = sentence_text.find(token.form, pos)
                if idx >= 0:
                    end = idx + len(token.form)
                    if end < len(sentence_text):
                        # Not at end of sentence - check for space
                        next_char = sentence_text[end]
                        space_after_range = bool(next_char and next_char.isspace())
                    # else: at end of sentence, leave as None (don't add SpaceAfter=No)
                    # advance pos past range form and any whitespace
                    pos = end
                    while pos < len(sentence_text) and sentence_text[pos].isspace():
                        pos += 1
            # For the last token (MWT range), never add SpaceAfter=No (set to None)
            if is_last_token:
                space_after_range = None
            
            end_id = current_id + len(token.subtokens) - 1
            form = _escape(token.form)
            misc_range = _format_misc_for_range(space_after_range)
            # MWT range line
            lines.append(f"{current_id}-{end_id}\t{form}\t_\t_\t_\t_\t_\t_\t_\t{misc_range}")
            # Subtokens: no SpaceAfter in MISC (they are not orthographic)
            for sub in token.subtokens:
                entity_label = token_entity_labels.get(current_id, "")
                lines.append(_format_token_line(current_id, sub, force_no_space=True, entity_label=entity_label, include_tokid=False, entity_format=entity_format, use_ne_format=use_ne_format, custom_misc_attrs=custom_misc_attrs))
                current_id += 1
        else:
            # Normal token: derive space from sentence_text or use explicit value
            space_after_tok: Optional[bool] = None
            # First, check if token has explicit space_after from input (e.g., from CoNLL-U)
            # This takes priority, especially for the last token
            if token.space_after is not None:
                space_after_tok = token.space_after
            elif sentence_text:
                idx = sentence_text.find(token.form, pos)
                if idx >= 0:
                    end = idx + len(token.form)
                    if end < len(sentence_text):
                        # Not at end of sentence - check for space
                        next_char = sentence_text[end]
                        space_after_tok = bool(next_char and next_char.isspace())
                    # else: at end of sentence, leave as None
                    # This allows _format_misc to check token.space_after if available
                    pos = end
                    while pos < len(sentence_text) and sentence_text[pos].isspace():
                        pos += 1
            # For the last token, never add SpaceAfter=No (set to None and ignore token's space_after)
            if is_last_token:
                space_after_tok = None
            
            entity_label = token_entity_labels.get(current_id, "")
            lines.append(_format_token_line(current_id, token, space_after_override=space_after_tok, ignore_token_space_after=is_last_token, entity_label=entity_label, include_tokid=False, entity_format=entity_format, use_ne_format=use_ne_format, custom_misc_attrs=custom_misc_attrs))
            current_id += 1

    return lines


def _format_token_line(
    token_id: int,
    token: Token | SubToken,
    *,
    space_after_override: Optional[bool] = None,
    force_no_space: bool = False,
    ignore_token_space_after: bool = False,
    entity_label: str = "",
    include_tokid: bool = False,
    entity_format: str = "iob",
    use_ne_format: bool = False,
    custom_misc_attrs: Optional[Dict[str, str]] = None,
) -> str:
    head = getattr(token, "head", 0) or 0
    head_value = "_" if head <= 0 else str(head)
    deprel = getattr(token, "deprel", "") or "_"
    deps = getattr(token, "deps", "") or "_"
    # If force_no_space is True (subtokens), suppress SpaceAfter completely (no entry in MISC)
    misc = _format_misc(
        token,
        space_after_override=(False if force_no_space else space_after_override),
        suppress_space_entry=force_no_space,
        ignore_token_space_after=ignore_token_space_after,
        entity_label=entity_label,
        custom_misc_attrs=custom_misc_attrs,
        include_tokid=include_tokid,
        entity_format=entity_format,
        use_ne_format=use_ne_format,
    )
    return (
        f"{token_id}\t{_escape(token.form)}\t{_escape(getattr(token, 'lemma', ''))}\t"
        f"{_escape(getattr(token, 'upos', ''))}\t{_escape(getattr(token, 'xpos', ''))}\t"
        f"{_escape(getattr(token, 'feats', ''))}\t{head_value}\t{_escape(deprel)}\t{_escape(deps)}\t{misc}"
    )


def _format_misc(
    token: Token | SubToken,
    *,
    space_after_override: Optional[bool] = None,
    suppress_space_entry: bool = False,
    ignore_token_space_after: bool = False,
    entity_label: str = "",
    include_tokid: bool = False,
    entity_format: str = "iob",
    use_ne_format: bool = False,
    custom_misc_attrs: Optional[Dict[str, str]] = None,
) -> str:
    entries: List[str] = []
    seen_keys: set[str] = set()
    misc_value = getattr(token, "misc", "")
    if misc_value and misc_value != "_":
        for part in misc_value.split("|"):
            if not part:
                continue
            # Extract key for deduplication
            if "=" in part:
                key = part.split("=", 1)[0]
            else:
                key = part
            # Skip SpaceAfter entries if ignore_token_space_after is True (for last token)
            # We'll handle SpaceAfter separately below
            if ignore_token_space_after and key == "SpaceAfter":
                continue
            # Deduplicate: keep first occurrence of each key
            if key not in seen_keys:
                seen_keys.add(key)
                entries.append(part)
    
    # Add NER entity label
    if entity_label:
        if use_ne_format:
            # NE= format: entity_label is already "ORG_3" format
            entries.append(f"NE={entity_label}")
        else:
            # IOB format: Entity=B-ORG, Entity=I-ORG, etc.
            entries.append(f"Entity={entity_label}")
    
    # Add additional attributes to MISC
    # Filter out "--" (reserved value in TEITOK) and "_" (empty value)
    reg = getattr(token, "reg", "")
    if reg and reg != "_" and reg != "--":
        entries.append(f"Normalization={reg}")
    mod = getattr(token, "mod", "")
    if mod and mod != "_" and mod != "--":
        entries.append(f"ModernForm={mod}")
    expan = getattr(token, "expan", "")
    if expan and expan != "_" and expan != "--":
        entries.append(f"Expansion={expan}")
    trslit = getattr(token, "trslit", "")
    if trslit and trslit != "_" and trslit != "--":
        entries.append(f"Translit={trslit}")
    ltrslit = getattr(token, "ltrslit", "")
    if ltrslit and ltrslit != "_" and ltrslit != "--":
        entries.append(f"LTransLit={ltrslit}")
    # TokId should be included if it came from original input, but suppressed if auto-generated by flexipipe
    # Auto-generated tokids follow the pattern s{num}-t{num} (e.g., s1-t1, s2-t5)
    # We only suppress these auto-generated ones by default
    tokid = getattr(token, "tokid", "")
    if tokid and tokid != "_":
        # Check if tokid matches the auto-generated pattern s{num}-t{num}
        is_auto_generated = bool(re.match(r'^s\d+-t\d+$', tokid))
        # Include if explicitly requested, or if it's not auto-generated (came from input)
        if include_tokid or not is_auto_generated:
            entries.append(f"TokId={tokid}")
    
    # Add custom attributes from attrs dictionary
    if custom_misc_attrs:
        attrs = getattr(token, "attrs", {})
        if attrs:
            for attr_key, misc_tag in custom_misc_attrs.items():
                if attr_key in attrs:
                    value = attrs[attr_key]
                    # Skip empty values, "_", and "--"
                    if value and value != "_" and value != "--":
                        # Escape the value if needed (CoNLL-U MISC values should not contain | or = in the value part)
                        # But we allow = in the key=value format, so we just need to escape | in the value
                        escaped_value = str(value).replace("|", "\\|")
                        tag_entry = f"{misc_tag}={escaped_value}"
                        # Deduplicate
                        if misc_tag not in seen_keys:
                            seen_keys.add(misc_tag)
                            entries.append(tag_entry)
    
    # Handle SpaceAfter
    # Priority:
    # 1. space_after_override (if not None) - from sentence_text derivation
    # 2. token.space_after (if not None and not ignore_token_space_after) - from input (e.g., CoNLL-U)
    # 3. None - unknown, don't add SpaceAfter entry (especially for last token)
    # Only add SpaceAfter=No if we have explicit False value
    space_flag: Optional[bool] = None
    if space_after_override is not None:
        # Explicit value from sentence_text derivation
        space_flag = space_after_override
    elif not ignore_token_space_after:
        # Fall back to token's space_after attribute (from input)
        # But skip this if ignore_token_space_after is True (e.g., for last token)
        space_flag_attr = getattr(token, "space_after", None)
        if space_flag_attr is not None:
            space_flag = space_flag_attr
        # else: leave as None (unknown), don't add SpaceAfter entry
    
    if not suppress_space_entry and space_flag is False:
        # Only add SpaceAfter=No if not already present (deduplicate)
        if "SpaceAfter" not in seen_keys:
            entries.append("SpaceAfter=No")
    return "|".join(entries) if entries else "_"


def _format_misc_for_range(space_after: Optional[bool]) -> str:
    if space_after is None:
        return "_"
    return "_" if space_after else "SpaceAfter=No"


def _reconstruct_sentence_text(tokens: Iterable[Token]) -> str:
    parts: List[str] = []
    for token in tokens:
        parts.append(token.form)
        if token.space_after:
            parts.append(" ")
    return "".join(parts).strip()


def _escape(value: str) -> str:
    return value if value else "_"


def _extract_misc_value(misc: str, key: str) -> str:
    if not misc or misc == "_":
        return ""
    prefix = f"{key}="
    for part in misc.split("|"):
        if part.startswith(prefix):
            value = part[len(prefix):]
            if value and value not in {"_", "--"}:
                return value
    return ""


def prepare_conllu_with_nlpform(source_path: Path, form_strategy: str) -> Path:
    """
    Produce a copy of a CoNLL-U file whose FORM column follows the requested strategy.

    Args:
        source_path: Original CoNLL-U file
        form_strategy: "form" (no change) or "reg" (use Normalization=... when present)
    """
    strategy = (form_strategy or "form").lower()
    if strategy == "form":
        return source_path
    if not source_path or not source_path.exists():
        return source_path

    suffix = f".nlpform-{strategy}.conllu"
    with source_path.open("r", encoding="utf-8") as src, tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=suffix, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        for line in src:
            stripped = line.rstrip("\n")
            if not stripped or stripped.startswith("#"):
                tmp.write(line)
                continue
            cols = stripped.split("\t")
            if len(cols) < 2:
                tmp.write(line)
                continue
            if strategy == "reg":
                misc_field = cols[9] if len(cols) > 9 else ""
                normalized = _extract_misc_value(misc_field, "Normalization")
                if normalized:
                    cols[1] = normalized
            tmp.write("\t".join(cols) + "\n")
    return tmp_path

