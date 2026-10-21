"""Placing sentences (<s>) into a tokenized TEITOK document by the corpus's element roles
(sentence_layout.py). Input: the backend's sentences as lists of <tok> elements in document order.

1. A sentence is cut where it crosses a ``hard`` edge (two tokens in different hard elements): that
   edge is a sentence end, whatever the backend said.
2. Each sentence is wrapped where it lies: around the children of the lowest element that holds all
   its tokens. Where it covers an element only partly, that element's crossing rule applies:
   ``split-element`` splits the element (a shell with the same attributes, without the id, takes the
   rest), ``split-s`` makes the part inside the element a separate <s> part (linked: ``part`` I/M/F,
   ``prev`` / ``next``), ``standoff`` gives up the element: the sentence becomes an empty <s> with
   its token list (``sameAs`` or the corpus's tokref attribute) before its first token.

Only elements move or are split: the text stays in place and in order (the writeback's text check
verifies it). Whitespace after a sentence's last token stays outside the <s>.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .sentence_layout import SentenceLayout


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _ns(el) -> str:
    tag = el.tag if isinstance(el.tag, str) else ""
    return tag[: tag.index("}") + 1] if tag.startswith("{") else ""


@dataclass
class PlacementStats:
    sentences: int = 0          # <s> made (a split sentence counts once)
    split_sentences: int = 0    # sentences in more than one part (split-s)
    parts: int = 0              # <s> elements of split sentences
    standoff: int = 0           # sentences as token lists
    cut_at_hard: int = 0        # backend sentences cut at a hard edge
    split_elements: Dict[str, int] = field(default_factory=dict)  # elements split, by name

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("sentences", "split_sentences", "parts", "standoff", "cut_at_hard", "split_elements")}


class _Placer:
    def __init__(self, root, layout: SentenceLayout, tokref_attr: str, id_prefix: str):
        self.root = root
        self.layout = layout
        self.tokref_attr = tokref_attr
        self.id_prefix = id_prefix
        self.stats = PlacementStats()
        self.counter = 0

    # ── tree helpers ─────────────────────────────────────────────────────
    @staticmethod
    def ancestors(el) -> List:
        out, p = [], el.getparent()
        while p is not None:
            out.append(p)
            p = p.getparent()
        return out

    def hard_ancestors(self, tok) -> List:
        # the elements themselves (compared with `is`): lxml makes a new proxy per access unless one
        # is held, so id() of a proxy is no identity of the node
        return [a for a in self.ancestors(tok) if self.layout.role(a).boundary == "hard"]

    @staticmethod
    def same_nodes(a: List, b: List) -> bool:
        return len(a) == len(b) and all(x is y for x, y in zip(a, b))

    def lca(self, toks):
        common = None
        for t in toks:
            chain = [t, *self.ancestors(t)]
            if common is None:
                common = chain
            else:
                ids = {id(x) for x in chain}
                common = [x for x in common if id(x) in ids]
        lca = common[0]
        return lca.getparent() if _local(lca.tag) == "tok" else lca

    @staticmethod
    def child_holding(container, el):
        while el is not None and el.getparent() is not container:
            el = el.getparent()
        return el

    @staticmethod
    def toks_in(el) -> List:
        if _local(el.tag) == "tok":
            return [el]
        return [t for t in el.iter() if _local(t.tag) == "tok"]

    # ── step 1: cut at hard edges ────────────────────────────────────────
    def cut(self, sentences: Sequence[Sequence]) -> List[List]:
        out = []
        for toks in sentences:
            toks = [t for t in toks if t is not None]
            if not toks:
                continue
            current = [toks[0]]
            hard = [self.hard_ancestors(t) for t in toks]  # held, so the proxies stay the same
            for i in range(1, len(toks)):
                prev, tok = toks[i - 1], toks[i]
                if not self.same_nodes(hard[i - 1], hard[i]):
                    out.append(current)
                    current = []
                    self.stats.cut_at_hard += 1
                current.append(tok)
            out.append(current)
        return out

    # ── splitting an element (split-element) ─────────────────────────────
    @staticmethod
    def _trailing_ws_out(el) -> None:
        """Whitespace at the end of `el`'s content moves out of it (into its tail): edge whitespace
        belongs outside an element (xmltokenizer's rule)."""
        holder = el[-1] if len(el) else None
        text = (holder.tail if holder is not None else el.text) or ""
        stripped = text.rstrip()
        if stripped == text:
            return
        if holder is not None:
            holder.tail = stripped or None
        else:
            el.text = stripped or None
        el.tail = text[len(stripped):] + (el.tail or "")

    def _detach_from(self, el, x):
        """Everything in `el` from `x` (a descendant) on, moved into a new detached shell of `el`
        (same tag and attributes, without the id); `el` keeps what comes before `x`."""
        shell = el.makeelement(el.tag, {k: v for k, v in el.attrib.items() if _local(k) != "id"})
        c = self.child_holding(el, x)
        if c is x:
            moving = [x, *list(x.itersiblings())]
        else:
            following = list(c.itersiblings())
            inner = self._detach_from(c, x)
            inner.tail, c.tail = c.tail, None
            self._trailing_ws_out(c)
            moving = [inner, *following]
        for m in moving:
            shell.append(m)  # moves it, with its tail
        return shell

    def split_before(self, el, x) -> None:
        """Split `el` in two siblings, the second starting at `x`; whitespace at the split goes between them."""
        name = _local(el.tag)
        self.stats.split_elements[name] = self.stats.split_elements.get(name, 0) + 1
        shell = self._detach_from(el, x)
        shell.tail, el.tail = el.tail, None
        self._trailing_ws_out(el)
        el.addnext(shell)

    # ── step 2: where the sentence's <s> elements go ─────────────────────
    def ranges(self, toks) -> Optional[List[Tuple]]:
        """[(container, first child, last child), …] in document order, or None for standoff."""
        mine = {id(t) for t in toks}
        role = lambda e: self.layout.role(e).crossing
        for _ in range(64):  # each pass splits at most one element
            container = self.lca(toks)
            kids = []
            for t in toks:
                k = self.child_holding(container, t)
                if not kids or kids[-1] is not k:
                    kids.append(k)
            partial = [e for e in {id(kids[0]): kids[0], id(kids[-1]): kids[-1]}.values()
                       if _local(e.tag) != "tok" and any(id(t) not in mine for t in self.toks_in(e))]
            k = next((e for e in partial if role(e) == "split-element"), None)
            if k is not None:  # split the element at the sentence's edge, then look again
                inside = [t for t in toks if self.child_holding(container, t) is k]
                if k is kids[0]:  # it starts before the sentence
                    self.split_before(k, inside[0])
                else:              # it goes on after the sentence: split at its first token after it
                    in_k = self.toks_in(k)
                    self.split_before(k, in_k[[id(t) for t in in_k].index(id(inside[-1])) + 1])
                continue
            elements = [e for e in kids if _local(e.tag) != "tok" and self.toks_in(e)]
            if any(role(e) == "standoff" for e in elements):
                return None
            # elements an <s> may not hold here: split-s ones, and wrap ones crossed partway
            descend = {id(e) for e in elements if role(e) == "split-s" or (role(e) == "wrap" and any(e is p for p in partial))}
            if not descend:
                return [(container, kids[0], kids[-1])]
            out: List[Tuple] = []
            run: List = []
            for e in kids:
                if id(e) in descend:
                    if run:
                        out.append((container, run[0], run[-1]))
                        run = []
                    inner = self.ranges([t for t in toks if self.child_holding(container, t) is e])
                    if inner is None:
                        return None
                    out.extend(inner)
                else:
                    run.append(e)
            if run:
                out.append((container, run[0], run[-1]))
            return out
        return None  # unexpected: leave the structure as it is (standoff)

    # ── writing ──────────────────────────────────────────────────────────
    def new_id(self) -> str:
        self.counter += 1
        return f"{self.id_prefix}{self.counter}"

    def wrap(self, container, first, last, sid: str):
        s = container.makeelement(_ns(container) + "s", {"id": sid})
        first.addprevious(s)
        node = first
        while node is not None:
            nxt = node.getnext()
            s.append(node)
            if node is last:
                break
            node = nxt
        # the text after the sentence's last token is outside the <s>
        s.tail, last.tail = last.tail, None
        return s

    def place(self, toks, text: Optional[str] = None) -> None:
        sid = self.new_id()
        ranges = self.ranges(toks)
        if ranges is None:
            ids = [t.get("id") or t.get("{http://www.w3.org/XML/1998/namespace}id") for t in toks]
            s = toks[0].makeelement(_ns(toks[0]) + "s", {"id": sid, self.tokref_attr: " ".join(f"#{i}" for i in ids if i)})
            if text:
                s.set("text", text)
            toks[0].addprevious(s)
            self.stats.standoff += 1
            self.stats.sentences += 1
            return
        parts = [self.wrap(c, f, l, sid if i == 0 else f"{sid}-{i + 1}") for i, (c, f, l) in enumerate(ranges)]
        if text:
            parts[0].set("text", text)
        if len(parts) > 1:
            for i, p in enumerate(parts):
                p.set("part", "I" if i == 0 else "F" if i == len(parts) - 1 else "M")
                if i > 0:
                    p.set("prev", "#" + parts[i - 1].get("id"))
                if i < len(parts) - 1:
                    p.set("next", "#" + parts[i + 1].get("id"))
            self.stats.split_sentences += 1
            self.stats.parts += len(parts)
        self.stats.sentences += 1


def place_sentences(
    root,
    sentences: Sequence[Sequence],
    layout: Optional[SentenceLayout] = None,
    *,
    texts: Optional[Sequence[Optional[str]]] = None,
    tokref_attr: str = "sameAs",
    id_prefix: str = "s-",
) -> PlacementStats:
    """Put <s> elements around the sentences (lists of <tok> elements in document order) by the
    element roles of `layout` (default: the written preset)."""
    placer = _Placer(root, layout or SentenceLayout(), tokref_attr, id_prefix)
    pieces = placer.cut(sentences)
    # a backend sentence cut at a hard edge keeps its text only when it was not cut
    whole = len(pieces) == len([s for s in sentences if s])
    for i, toks in enumerate(pieces):
        placer.place(toks, (texts[i] if texts and whole and i < len(texts) else None))
    return placer.stats
