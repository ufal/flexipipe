# -*- coding: utf-8 -*-
"""
TEITOK-style XML tokenizer (Python port of xmltokenize.pl).

Tokenizes the content of a TEI <text> (or similar) element by:
- Splitting on whitespace and wrapping runs in <tok>
- Splitting off leading/trailing punctuation from tokens
- Moving unbalanced XML tags out of token boundaries (or adding rpt="1" to repair)
- Optionally splitting sentences after [.?!] and wrapping in <s>

This strategy works on the serialized string (regex/line-by-line) and has been
adapted to TEITOK/XML over a long time; it seldom fails except for large stacks
of elements without spaces (often malformed from TEITOK's perspective).

Complex cases (many tags split off tokens or duplicated with rpt=\"1\"): the
same logic as the Perl script is used (move tags at boundaries, peel unbalanced
tags, repair with rpt=\"1\"), so behaviour should be equivalent.

Use --insert-tokens-engine teitok to select this path without altering the
standoff-based approach.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Optional XML parsing for validation
try:
    import xml.etree.ElementTree as ET
except ImportError:
    ET = None  # type: ignore


# Block elements: add newline before so they are on their own "line"
# TODO: These can differ per corpus; should be configurable via CLI/teitok-settings (pelms, notok, flush, etc.)
DEFAULT_BLOCK_ELEMENTS = ["div", "head", "p", "u", "speaker"]
# Elements to extract (replace with placeholder so they are not tokenized)
DEFAULT_NOTOK_ELEMENTS = "note|desc|gap|pb|fw|rdg"


def _tag_name(tag: str) -> str:
    """Get local name from possibly namespaced tag."""
    return tag.split("}")[-1].split("/")[-1].lower()


def _is_punctuation(ch: str) -> bool:
    if not ch:
        return False
    if ch in " \t\n\r":
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("P") or ch in ".,;:!?\"'„‟‚‛″′‹›«»()[]{}…-–—"


def _extract_text_region(
    raw_xml: str,
    text_tag: str = "text",
) -> Tuple[str, str, str]:
    """
    Split XML into head, content of first <text>...</text>, and foot.
    Returns (head, tagtxt, foot).
    """
    # Match first <text> or <text ...> through to </text>
    pattern = re.compile(
        r"(<" + re.escape(text_tag) + r"(?:\s[^>]*)?>)(.*?)(</" + re.escape(text_tag) + r">)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(raw_xml)
    if not m:
        raise ValueError(f"No element <{text_tag}> found in XML")
    open_tag = m.group(1)
    content = m.group(2)
    close_tag = m.group(3)
    head = raw_xml[: m.start()]
    foot = raw_xml[m.end() :]
    tagtxt = open_tag + content + close_tag
    return head, tagtxt, foot


def _remove_linebreaks_inside_tags(tagtxt: str, max_passes: int = 5) -> str:
    """Replace newlines inside a tag with space so tags stay on one line."""
    for _ in range(max_passes):
        new_txt = re.sub(r"<([^>\n\r]*?)[\n\r]+\s*", r"<\1 ", tagtxt)
        if new_txt == tagtxt:
            break
        tagtxt = new_txt
    return tagtxt


def _protect_entities(tagtxt: str) -> str:
    """Protect &amp; &lt; &gt; and &entity; so they are not split."""
    s = tagtxt.replace("&amp;", "xxAMPxx").replace("&lt;", "xxLTxx").replace("&gt;", "xxGTxx")
    # Protect &entity;
    s = re.sub(r"(&[^ \n\r&]+;)", r"xx\1xx", s)
    s = re.sub(r"&(?![^ \n\r&;]+;)", "xx&amp;xx", s)
    return s


def _unprotect_entities(teitext: str) -> str:
    s = re.sub(r"xx(&(?!xx)[^ \n\r&]+;)xx", r"\1", teitext)
    s = s.replace("xxAMPxx", "&amp;").replace("xxLTxx", "&lt;").replace("xxGTxx", "&gt;")
    return s


def _extract_notes(
    tagtxt: str,
    notok_pattern: str,
) -> Tuple[str, List[str]]:
    """
    Replace elements matching notok_pattern (and XML comments) with <ntn n="N"/>.
    Returns (modified_tagtxt, notes_list).
    """
    notes: List[str] = []
    # Match <note ...>...</note> etc. (non-nested by this regex)
    pattern = re.compile(
        r"<(" + notok_pattern + r")[^>]*(?<!/)>.*?</\1>",
        re.IGNORECASE | re.DOTALL,
    )
    prev = None
    loop_guard = 0
    while True:
        m = pattern.search(tagtxt)
        if not m:
            break
        notetxt = m.group(0)
        left = tagtxt[: m.start()]
        key = left[-50:] + "#" + notetxt
        if key == prev:
            loop_guard += 1
            if loop_guard > 5:
                break
        else:
            loop_guard = 0
        prev = key
        notes.append(notetxt)
        tagtxt = tagtxt[: m.start()] + f'<ntn n="{len(notes)-1}"/>' + tagtxt[m.end() :]

    # XML comments
    pattern_comment = re.compile(r"<!--.*?-->", re.DOTALL)
    while True:
        m = pattern_comment.search(tagtxt)
        if not m:
            break
        notetxt = m.group(0)
        notes.append(notetxt)
        tagtxt = tagtxt[: m.start()] + f'<ntn n="{len(notes)-1}"/>' + tagtxt[m.end() :]

    return tagtxt, notes


def _restore_notes(teitext: str, notes: List[str]) -> str:
    for i, notetxt in enumerate(notes):
        placeholder = f'<ntn n="{i}"/>'
        teitext = teitext.replace(placeholder, notetxt)
    return teitext


def _add_newlines_before_blocks(tagtxt: str, block_names: str) -> str:
    """Add newline before block-opening tags so each block is on its own line."""
    # Match full opening tag: <tag or <tag attr="...">
    pattern = r"(?<![\n\r])(<(?:" + block_names + r")(?=[ >])[^>]*>)"
    return re.sub(pattern, r"\n\1", tagtxt)


def _tokenize_line(line: str, tok_counter: Optional[List[int]] = None) -> str:
    """
    Tokenize a single line: split on whitespace, wrap in <tok>, split punctuation,
    fix unbalanced tags inside each token. tok_counter is [next_id] shared across lines.
    """
    if tok_counter is None:
        tok_counter = [1]
    if not line.strip():
        return line

    # Preserve " xml:" and " attr=" inside tags so protect/restore don't corrupt them
    _XML_COLON = "xxXMLCOLONxx"
    line = line.replace(" xml:", _XML_COLON)
    # Only inside tags: " id=" -> placeholder (so we don't get " id =" -> " id=" later)
    line = re.sub(r"(<[^>]*) id=", r"\1xxIDEQxx", line)

    # Protect all spaces inside XML tags so splitting on whitespace doesn't break tags
    # (e.g. <head type="h2"> has space between "head" and "type" that must not be split)
    line = re.sub(r"<([^>]*)>", lambda m: "<" + m.group(1).replace(" ", "%%") + ">", line)

    pre = ""
    post = ""
    m = re.match(r"^(\s*)", line)
    if m:
        pre = m.group(1)
        line = line[m.end() :]
    m = re.search(r"(\s*)$", line)
    if m:
        post = m.group(1)
        line = line[: m.start()]

    if not line:
        return pre + post

    # Wrap in placeholder and split on whitespace
    line = "<tokk>" + line + "</tok>"
    line = re.sub(r"(\s+)", r"</tok>\1<tokk>", line)

    # Remove tokens that contain only XML tags
    line = re.sub(r"<tokk>((?:<[^>]+>)+)</tok>", r"\1", line)

    # Split at same-tag boundary (e.g. "word</item><item>") so "Primary" and "Key" stay separate runs.
    # Do not split when </tag> is preceded by whitespace or ">" (e.g. "<s/></item><item>" or " </item><item>"), else we get an extra </tok>.
    for tag in ("item", "list", "hi"):
        line = re.sub(
            r"(?<![ \n\t>])(</" + tag + r">)(<" + tag + r"(?:\s[^>]*)?>)",
            r"\1</tok><tokk>\2",
            line,
        )

    # Split runs at "text)(</tag" only when remainder has an opening tag (e.g. "</hi></item><item>Into");
    # do not split ".(</item></list>)" else we get incomplete run "<tokk></item></list>" with no "</tok>"
    def _split_text_before_structural(m: re.Match) -> str:
        content = m.group(2)
        def replacer(m2: re.Match) -> str:
            # Only split if after this "</" there is an opening tag <tagname (not </)
            after = content[m2.end() :]
            if re.search(r"<[^/]", after):
                return m2.group(1) + "</tok><tokk>" + m2.group(2)
            return m2.group(0)
        content = re.sub(r"([.)\]}\s])(</)", replacer, content)
        return m.group(1) + content + m.group(3)

    line = re.sub(
        r"(<tokk>)(.*?)(</tok>)",
        _split_text_before_structural,
        line,
        flags=re.DOTALL,
    )

    # Restore %% to space; fix spurious spaces (attr = and xml :)
    line = line.replace("%%", " ")
    line = re.sub(r"xml :", "xml:", line)
    # Fix namespace-prefixed attributes (e.g. " xml: lang =" -> " xml:lang="); else " lang =" gets fixed and leaves invalid " xml: lang="
    line = re.sub(r" ([a-zA-Z][a-zA-Z0-9]*):\s*([a-zA-Z][a-zA-Z0-9]*)\s*=", r" \1:\2=", line)
    line = re.sub(r" ([a-zA-Z][a-zA-Z0-9]*) =", r" \1=", line)
    line = line.replace("xxXMLCOLONxx", " xml:")
    line = line.replace("xxIDEQxx", " id=")

    # Move closing tags at end of token content out first (so "word.</p></tok>" -> "word.</tok></p>")
    # so that punct split can then split "."
    for _ in range(5):
        prev = line
        line = re.sub(r"<tokk>(.*?)(</[^>]+>)</tok>", r"<tokk>\1</tok>\2", line, flags=re.DOTALL)
        if line == prev:
            break
    # Move opening tags at end of token out
    line = re.sub(r"(<[^/][^>]*>)</tok>", r"</tok>\1", line)
    # Move closing tags at start of token out
    line = re.sub(r"<tokk>(</[^>]+>)", r"\1<tokk>", line)
    # Move <ntn> and <p> out
    line = re.sub(r"<tokk>(<ntn [^>]+/>)", r"\1<tokk>", line)
    line = re.sub(r"(<ntn [^>]+/>)</tok>", r"</tok>\1", line)
    line = re.sub(r"<tokk>(<p [^>]*>)", r"\1<tokk>", line)
    line = re.sub(r"(<p [^>]*>)</tok>", r"</tok>\1", line)

    # Move punctuation: (punct)(<tag>)</tok> -> (punct)</tok>(<tag>)
    line = re.sub(r"([^\s<])(<[^>]+>)</tok>", r"\1</tok>\2", line)
    # <tokk>(<tag>)(punct) -> (<tag>)<tokk>(punct)
    line = re.sub(r"<tokk>(<[^>]+>)([^\s>])", r"\1<tokk>\2", line)

    # Split off leading/trailing punctuation from tokens (repeat until no change)
    for _ in range(10):
        prev = line
        line = re.sub(r"([^\s<])</tok>", _split_trailing_punct_right, line)
        line = re.sub(r"<tokk>([^\s<])", _split_leading_punct_right, line)
        if line == prev:
            break
    line = re.sub(r"<tokk></tok>", "", line)
    line = re.sub(r"<tokk>(<[^>]+>)</tok>", r"\1", line)

    # Move tags at boundaries (again after punct split)
    line = re.sub(r"(<[^/][^>]*>)</tok>", r"</tok>\1", line)
    line = re.sub(r"<tokk>(</[^>]+>)", r"\1<tokk>", line)
    # Move <ntn> and <p> out
    line = re.sub(r"<tokk>(<ntn [^>]+/>)", r"\1<tokk>", line)
    line = re.sub(r"(<ntn [^>]+/>)</tok>", r"</tok>\1", line)
    line = re.sub(r"<tokk>(<p [^>]*>)", r"\1<tokk>", line)
    line = re.sub(r"(<p [^>]*>)</tok>", r"</tok>\1", line)

    # If same-tag split left <tokk><tag> without </tok> (at line end or before another <tok>), close the run so replace_tok can process it
    for tag in ("item", "list", "hi"):
        line = re.sub(r"(<tokk><" + tag + r"(?:\s[^>]*)?>)(\s*)$", r"\1</tok>\2", line)
        line = re.sub(r"(<tokk><" + tag + r"(?:\s[^>]*)?>)(\s*)(<tok\s)", r"\1</tok>\2\3", line)

    # Process each <tokk>...</tok> and ensure valid XML; assign id
    def replace_tok(m: re.Match) -> str:
        content = m.group(1)
        a, inner, b = _fix_token_content(content)
        tok_id = tok_counter[0]
        tok_counter[0] += 1
        return a + f'<tok id="w-{tok_id}">' + inner + "</tok>" + b

    line = re.sub(r"<tokk>(.*?)</tok>", replace_tok, line, flags=re.DOTALL)

    # Stray </hi> after <list><item> (parser inside <item>) — move </hi> before <list><item> so we close hi first
    line = re.sub(r"(<list><item>)</hi>\s*(<tok)", r"</hi>\1\2", line)
    line = re.sub(r"(</item><list><item>)</hi>\s*(<tok)", r"</hi>\1\2", line)
    # Space-only run before <s/> produced " </tok><s/>" or "</tok><s/></item>..." which leaves stray </tok>; collapse so we don't emit extra </tok>
    line = re.sub(r" </tok><s/>", "<s/>", line)
    line = re.sub(r" </tok><s/></item>", "<s/></item>", line)
    line = re.sub(r"</tok><s/></item>", "<s/></item>", line)
    line = re.sub(r"<s/></tok></item>", "<s/></item>", line)

    # Join [...] and (...)
    line = re.sub(
        r"<tok[^>]*>\[</tok><tok[^>]*>\.</tok><tok[^>]*>\.</tok><tok[^>]*>\.</tok><tok[^>]*>\]</tok>",
        "<tok>[...]</tok>",
        line,
    )
    line = re.sub(
        r"<tok[^>]*>\.</tok><tok[^>]*>\.</tok><tok[^>]*>\.</tok>",
        "<tok>...</tok>",
        line,
    )

    return pre + line + post


def _split_trailing_punct_right(m: re.Match) -> str:
    """If last char before </tok> is punct, move it into its own token."""
    ch = m.group(1)
    if _is_punctuation(ch):
        return "</tok><tokk>" + ch + "</tok>"
    return m.group(0)


def _split_leading_punct_right(m: re.Match) -> str:
    """If first char is punct, split it off into its own token (close current, open new)."""
    ch = m.group(1)
    if _is_punctuation(ch):
        return "<tokk>" + ch + "</tok><tokk>"
    return m.group(0)


def _fix_token_content(content: str) -> Tuple[str, str, str]:
    """
    Ensure content inside <tok>...</tok> is valid XML.
    Peel unbalanced tags to prefix (a) and suffix (b), or add rpt="1" to repair.
    Returns (a, inner, b).
    """
    a = ""
    b = ""
    inner = content
    peeled_to_a: List[str] = []  # tag names we peeled to (a); close them in LIFO order in (b)
    if not ET:
        return a, inner, b

    def close_peeled() -> None:
        nonlocal b
        # Only close the innermost peeled tag so we don't close e.g. <list> after first <item>;
        # outer tags will be closed when a later run emits their </list> etc.
        # If (a) already ends with "</tagname>" we're already closing that tag, so don't add again.
        # If (a) contains a new opening of the same tag (e.g. "</item><item>") don't add "</item>"
        # or we'd close the item we just opened; it will be closed when a later run has "</item>" in (a).
        # When (a) is only opening tags (e.g. "<list><item>"), only add closing for tags opened in (a)
        # so we don't output </hi> when parser is inside <item>.
        if peeled_to_a:
            innermost = peeled_to_a[-1]
            a_stripped = a.lstrip()
            if a_stripped.startswith("<") and not a_stripped.startswith("</"):
                open_in_a = set()
                for tag_m in re.finditer(r"<([a-zA-Z][a-zA-Z0-9:]*)(?:\s|[>])", a):
                    open_in_a.add(tag_m.group(1).split("}")[-1].lower())
                if innermost not in open_in_a:
                    return
            if a.rstrip().endswith("</" + innermost + ">"):
                return
            if re.search(r"<" + re.escape(innermost) + r"(?:\s[^>]*)?>", a):
                return
            b = "</" + innermost + ">" + b

    for _ in range(20):
        try:
            ET.fromstring(f"<tok>{inner}</tok>")
            # Don't leave opening tags inside <tok> when they have no matching close in inner
            # (e.g. <item>Into -> peel <item> to prefix so we get ...<item><tok>Into</tok>)
            rest = inner.lstrip()
            if rest.startswith("<") and not rest.startswith("</"):
                open_m = re.match(r"^<([a-zA-Z][a-zA-Z0-9:]*)", rest)
                if open_m:
                    tn = open_m.group(1).split("}")[-1].lower()
                    full_tag_m = re.match(r"^<[^>]+>", rest)
                    if full_tag_m and not re.search(r"</" + re.escape(tn) + r"[>\s]", inner):
                        # Opening tag at start with no matching close in inner -> peel to (a)
                        prefix_len = len(inner) - len(rest)
                        a += inner[:prefix_len] + full_tag_m.group(0)
                        peeled_to_a.append(tn)
                        inner = rest[len(full_tag_m.group(0)) :]
                        continue
            # Never emit our placeholder "</tok>" from (a) - parser would see it and expect we're closing a tok
            if "</tok>" in a:
                a = a.replace("</tok>", "")
            # If (a) is only a single closing tag and inner is non-empty, dropping (a) avoids e.g. </hi><tok>word</tok> when parser is inside <item>
            if inner.strip() and re.match(r"^</[a-zA-Z][a-zA-Z0-9:]*>$", a.strip()):
                a = ""
            close_peeled()
            return a, inner, b
        except ET.ParseError:
            pass

        # Peel leftmost: self-closing, closing, or opening without match
        if re.match(r"^<([^>]+)/>", inner):
            m = re.match(r"^<[^>]+/>", inner)
            if m:
                a += m.group(0)
                inner = inner[m.end() :]
                continue
        if re.match(r"^</[^>]+>", inner):
            m = re.match(r"^</([a-zA-Z][a-zA-Z0-9:]*)>", inner)
            if m and m.group(1).split("}")[-1].lower() != "tok":
                # If (a) ends with an opening tag, this closing tag may be stray (e.g. "<list><item></hi>")
                a_rstrip = a.rstrip()
                if a_rstrip.endswith(">"):
                    last_lt = a_rstrip.rfind("<")
                    if last_lt >= 0 and last_lt + 1 < len(a_rstrip) and a_rstrip[last_lt + 1] != "/":
                        inner = inner[m.end() :]
                        continue
                # If (a) contains an opening <list> or <item>, don't emit a different closing (e.g. </hi>) — drop stray (e.g. "</item><list><item></hi>")
                if re.search(r"<(?:list|item)(?:\s|[>])", a):
                    inner = inner[m.end() :]
                    continue
                a += m.group(0)
                inner = inner[m.end() :]
                continue
            # "</tok>" is our placeholder from split - strip leading "</tok><tokk>" and use rest as inner
            if inner.startswith("</tok><tokk>"):
                inner = inner[len("</tok><tokk>") :]
                continue
            if inner.startswith("</tok>"):
                inner = inner[len("</tok>") :]
                continue
        if re.match(r"^<[^/][^>]*>", inner):
            m = re.match(r"^<[^/][^>]*>", inner)
            tag = m.group(0)
            tname = re.match(r"^<([a-zA-Z0-9:]+)", tag)
            tn = tname.group(1).split("}")[-1] if tname else ""
            # Check if there is a matching close later
            if tn and re.search(r"</" + re.escape(tn) + r"[>\s]", inner[m.end() :]):
                a += tag
                inner = inner[m.end() :]
                continue
            # No match, peel
            a += tag
            peeled_to_a.append(tn.lower())
            inner = inner[m.end() :]
            continue
        if re.search(r"<[^>]+/>\s*$", inner):
            m = re.search(r"<[^>]+/>\s*$", inner)
            if m:
                b = m.group(0) + b
                inner = inner[: m.start()]
                continue
        if re.search(r"</[^>]+>\s*$", inner):
            m = re.search(r"</([a-zA-Z][a-zA-Z0-9:]*)>\s*$", inner)
            if m and m.group(1).split("}")[-1].lower() != "tok":
                b = m.group(0) + b
                inner = inner[: m.start()]
                continue
        # Do not peel opening tags from the right to (b) - they have no matching close; mid-inner peel will move them to (a)

        # Strip our placeholder "<tokk>" from the start of inner (from bad split)
        if inner.lstrip().startswith("<tokk>"):
            m_tokk = re.match(r"^(\s*)<tokk>", inner)
            if m_tokk:
                inner = inner[m_tokk.end() :]
                continue

        # Peel every opening tag that has no matching close (outermost first so (a) order is e.g. "<list><item>");
        # close_peeled() only adds innermost closing so we don't close <list> after first <item>
        did_peel = False
        while True:
            open_mid = re.search(r"<([a-zA-Z][a-zA-Z0-9:]*)(?:\s[^>]*)?>", inner)
            if not open_mid or open_mid.group(1).split("}")[-1].lower() in ("tokk", "tok"):
                break
            full_open = open_mid.group(0)
            tn = open_mid.group(1).split("}")[-1].lower()
            if not re.search(r"</" + re.escape(tn) + r"[>\s]", inner):
                a += full_open
                peeled_to_a.append(tn)
                inner = inner[: open_mid.start()] + inner[open_mid.end() :]
                did_peel = True
            else:
                break
        if did_peel:
            continue

        # Count tags and repair with rpt="1"; track open_order so we close in LIFO (document order)
        open_count: Dict[str, int] = {}
        open_order: List[str] = []  # order openings appear in inner, for correct closing order
        rpt_added_to_inner: List[str] = []  # tags we inserted with rpt="1" in inner; need their closings in (b) then inside token
        i = 0
        while i < len(inner):
            if inner[i] == "<":
                close = inner.find(">", i)
                if close == -1:
                    break
                tag = inner[i : close + 1]
                is_closing = tag.startswith("</")
                is_self = tag.endswith("/>")
                if is_self:
                    i = close + 1
                    continue
                tname = re.match(r"</?([a-zA-Z0-9:]+)", tag)
                tn = tname.group(1).split("}")[-1].lower() if tname else ""
                if not tn:
                    i = close + 1
                    continue
                if is_closing:
                    open_count[tn] = open_count.get(tn, 0) - 1
                    if open_count[tn] < 0:
                        a += tag
                        inner = inner[:i] + f"<{tn} rpt=\"1\">" + inner[close + 1 :]
                        open_count[tn] = 0
                        rpt_added_to_inner.append(tn)
                        continue
                    # Remove one matching open from the end of open_order (LIFO)
                    for j in range(len(open_order) - 1, -1, -1):
                        if open_order[j] == tn:
                            open_order.pop(j)
                            break
                else:
                    open_count[tn] = open_count.get(tn, 0) + 1
                    open_order.append(tn)
                i = close + 1
            else:
                i += 1
        _STRUCTURAL_TAGS = {"item", "list", "hi", "head", "p", "div", "u", "speaker"}
        # Close in LIFO order (open_order is push order, so reversed = close order)
        closings_from_open = ["</" + tn + ">" for tn in reversed(open_order)]
        # Closings for rpt="1" we added to inner must be in (b) so "Put closing tags inside token" can move them in
        rpt_set = set(rpt_added_to_inner)
        closings = ["</" + t + ">" for t in rpt_added_to_inner] + [c for c in closings_from_open if not (m := re.match(r"</([a-zA-Z][a-zA-Z0-9:]*)>", c)) or m.group(1).split("}")[-1].lower() not in rpt_set]
        # When (a) is only opening tags (e.g. "<list><item>"), don't add closings for other tags (e.g. </hi>)
        # unless they close an rpt="1" we added to inner (those go inside the token)
        a_stripped = a.lstrip()
        if a_stripped.startswith("<") and not a_stripped.startswith("</"):
            open_in_a = set()
            for tag_m in re.finditer(r"<([a-zA-Z][a-zA-Z0-9:]*)(?:\s|[>])", a):
                open_in_a.add(tag_m.group(1).split("}")[-1].lower())
            rpt_set = set(rpt_added_to_inner)
            closings = [
                c for c in closings
                if (m := re.match(r"</([a-zA-Z][a-zA-Z0-9:]*)>", c))
                and (m.group(1).split("}")[-1].lower() in open_in_a or m.group(1).split("}")[-1].lower() in rpt_set)
            ]
        b = "".join(closings) + b
        for tn in open_order:
            if tn.lower() not in _STRUCTURAL_TAGS:
                inner = inner + ("<" + tn + ' rpt="1">')
        # For structural tags we only added closing to (b), not opening to inner
        # Put closing tags (b) inside the token when we added rpt="1" openings to inner
        if b and "<" in inner and " rpt=\"1\">" in inner:
            inner = inner + b
            b = ""
        break

    # Add closing for innermost peeled tag only (avoids closing <list> after first <item>)
    close_peeled()
    return a, inner, b


def _sentence_split(teitext: str, block_names: str) -> str:
    """
    Insert <s/> after block opens and after [.?!]</tok>; then normalize.
    block_names: pipe-separated block element names (e.g. "div|head|p|u|speaker").
    """
    # <s/> after full block opening tag
    full_tag = r"<(?:" + block_names + r")(?=[ >])[^>]*>"
    teitext = re.sub(r"(" + full_tag + r")", r"\1\n<s/>", teitext)
    # <s/> after .?!</tok>
    teitext = re.sub(r"(<tok[^>]*>[.?!]</tok>)(\s*)", r"\1\2\n<s/>", teitext)
    # Remove <s/> before </block>
    teitext = re.sub(r"<s/>(</(?:" + block_names + r")>)", r"\1", teitext)
    return teitext


def tokenize_teitok_style(
    raw_xml: str,
    text_tag: str = "text",
    block_elements: Optional[List[str]] = None,
    notok_elements: Optional[str] = None,
    split_sentences: bool = True,
    keep_ns: bool = False,
) -> str:
    """
    Tokenize XML using TEITOK's string/regex strategy.

    Args:
        raw_xml: Full XML file content (or at least the <text> region).
        text_tag: Element name that contains the body to tokenize (default "text").
        block_elements: Tags before which to add newlines (default div, head, p, u, speaker).
        notok_elements: Pipe-separated tag names to replace with placeholders (default note|desc|gap|pb|fw|rdg).
        split_sentences: If True, insert <s/> after sentence-ending punctuation and block opens.
        keep_ns: If False, replace xmlns= with xmlnsoff= to avoid namespace issues.

    Returns:
        Full XML string with <tok> (and optionally <s/>) inserted inside the text region.
    """
    if block_elements is None:
        block_elements = DEFAULT_BLOCK_ELEMENTS
    if notok_elements is None:
        notok_elements = DEFAULT_NOTOK_ELEMENTS

    if not keep_ns:
        raw_xml = raw_xml.replace(" xmlns=", " xmlnsoff=")

    head, tagtxt, foot = _extract_text_region(raw_xml, text_tag)
    tagtxt = _remove_linebreaks_inside_tags(tagtxt)

    tagtxt, notes = _extract_notes(tagtxt, notok_elements)
    tagtxt = _protect_entities(tagtxt)

    block_reg = "|".join(re.escape(e) for e in block_elements)
    block_pattern = r"<(" + block_reg + r")[ >/]"  # for sentence split
    tagtxt = _add_newlines_before_blocks(tagtxt, block_reg)

    lines = tagtxt.split("\n")
    tok_counter: List[int] = [1]
    teitext_lines: List[str] = []
    for line in lines:
        teitext_lines.append(_tokenize_line(line, tok_counter))
    teitext = "\n".join(teitext_lines)

    teitext = _unprotect_entities(teitext)
    if split_sentences:
        teitext = _sentence_split(teitext, block_reg)
    # Clean stray </tok> after <s/> that cause "mismatched tag" (e.g. <s/></tok></item>)
    teitext = re.sub(r"<s/></tok></item>", "<s/></item>", teitext)
    teitext = _restore_notes(teitext, notes)

    out = head + teitext + foot
    # Fix spurious spaces from %% restore: "xml :" -> "xml:", " xml: lang =" -> " xml:lang=", " attr =" -> " attr="
    out = re.sub(r"xml :", "xml:", out)
    out = re.sub(r" ([a-zA-Z][a-zA-Z0-9]*):\s*([a-zA-Z][a-zA-Z0-9]*)\s*=", r" \1:\2=", out)
    out = re.sub(r" ([a-zA-Z][a-zA-Z0-9]*) =", r" \1=", out)
    return out


def rebuild_teitok_xml_with_tokens(
    full_xml: str,
    document: Any,
    block_elements: Optional[List[str]] = None,
    notok_elements: Optional[str] = None,
    split_sentences: bool = True,
) -> str:
    """
    Tokenize the XML using TEITOK-style, then merge token IDs and attributes
    from the backend Document (by order) so that each <tok> gets id, form, lemma, etc.

    full_xml: Original XML string.
    document: flexipipe Document with sentences/tokens (from backend).
    """
    tokenized = tokenize_teitok_style(
        full_xml,
        text_tag="text",
        block_elements=block_elements,
        notok_elements=notok_elements,
        split_sentences=split_sentences,
    )
    # Optional: walk tokenized XML and assign document token attributes by order
    # For now we leave the tokenized string as-is (ids are w-1, w-2, ...).
    return tokenized
