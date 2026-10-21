"""How sentences (<s>) relate to the elements of a document: a role per element, chosen per corpus.

`<u>` is a speaker turn in a conversation corpus and whatever fits on a screen in a subtitle corpus;
`<seg>` (timed) often ends mid-sentence. So each element gets two settings:

- boundary — is the element's edge a sentence end? ``hard`` (always: no sentence crosses it),
  ``cue`` (evidence for the segmenter; for placement the same as ``none``), ``none``;
- crossing — what happens when a sentence goes beyond the element: ``split-s`` (the element is never
  inside an <s> that goes beyond it: the <s> is split into linked parts inside it; for elements with
  timing or identity: u, seg), ``split-element`` (the element is split around the <s>; inline
  elements), ``standoff`` (no <s> element: the sentence is a token list), ``wrap`` (an <s> may contain
  the element whole; crossed partway, as split-s — the default for elements not in the preset, e.g.
  <name>).

Presets cover the usual corpora; a corpus overrides single elements. Configured in an application
profile (``segmentation: {layout: {preset, elements: {seg: {boundary, crossing}}}}``), in settings.xml
(``<flexipipe><segmentation preset="conversation"><element name="seg" type="pw" crossing="wrap"/>
</segmentation>``: a role for one @type of an element, ``seg[type=pw]`` in a profile) or with
``--sentence-layout PRESET``.
Plan: dev/SPOKEN-PRONUNCIATION-AND-SENTENCES.md (phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

BOUNDARIES = ("hard", "cue", "none")
CROSSINGS = ("split-s", "split-element", "standoff", "wrap")

# structural (block) elements: a sentence never crosses them in any preset
BLOCKS = ("text", "body", "front", "back", "div", "p", "ab", "head", "l", "lg", "list", "item",
          "table", "row", "cell", "note", "figure", "quote", "sp", "speaker", "stage", "trailer")
# inline elements without identity of their own: split around a sentence
INLINE = ("hi", "emph", "foreign", "del", "add", "ex", "expan", "abbr", "sic", "corr", "orig", "reg",
          "unclear", "supplied", "mod", "subst", "choice", "c")


@dataclass(frozen=True)
class Role:
    boundary: str = "none"
    crossing: str = "split-s"

    def __post_init__(self):
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"sentence layout: boundary '{self.boundary}' (choose from {', '.join(BOUNDARIES)})")
        if self.crossing not in CROSSINGS:
            raise ValueError(f"sentence layout: crossing '{self.crossing}' (choose from {', '.join(CROSSINGS)})")


def _preset(**extra: Role) -> Dict[str, Role]:
    roles = {name: Role("hard", "split-s") for name in BLOCKS}
    roles.update({name: Role("none", "split-element") for name in INLINE})
    roles.update(extra)
    return roles


PRESETS: Dict[str, Dict[str, Role]] = {
    "written": _preset(),
    # u = a real speaker turn; seg = timed segments inside it, often ending mid-sentence
    "conversation": _preset(u=Role("hard", "split-s"), seg=Role("cue", "split-s")),
    # u = a subtitle screen: sentences run across screens; the <s> is split, the screen stays whole
    "subtitles": _preset(u=Role("cue", "split-s"), seg=Role("cue", "split-s")),
    # the same without <s> fragments: a sentence across screens is a token list
    "subtitles-standoff": _preset(u=Role("cue", "standoff"), seg=Role("cue", "standoff")),
}
DEFAULT_PRESET = "written"
# an element in no preset: may be inside an <s> whole, never split (crossed partway: split-s)
UNKNOWN = Role("none", "wrap")


@dataclass
class SentenceLayout:
    preset: str = DEFAULT_PRESET
    overrides: Dict[str, Role] = field(default_factory=dict)

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"sentence layout: unknown preset '{self.preset}' (choose from {', '.join(PRESETS)})")

    def role(self, element, type_: Optional[str] = None) -> Role:
        """The role of an element (an lxml element, or a tag name and its @type): a role keyed
        ``name[type=…]`` wins over ``name`` (ORTOFON: seg[type=pw] vs seg[type=overlap])."""
        if not isinstance(element, str):
            type_ = element.get("type")
            element = element.tag if isinstance(element.tag, str) else ""
        name = element.rsplit("}", 1)[-1]
        preset = PRESETS[self.preset]
        keys = ([f"{name}[type={type_}]"] if type_ else []) + [name]
        for key in keys:
            if key in self.overrides:
                return self.overrides[key]
            if key in preset:
                return preset[key]
        return UNKNOWN

    def describe(self) -> Dict[str, Dict[str, str]]:
        """The roles that differ from an unknown element's, for status pages."""
        # the elements this layout is about (u, seg, …) and the corpus's own settings; not the block
        # and inline elements every preset treats alike
        roles = {n: r for n, r in PRESETS[self.preset].items() if n not in BLOCKS and n not in INLINE}
        roles.update(self.overrides)
        return {n: {"boundary": r.boundary, "crossing": r.crossing} for n, r in sorted(roles.items())}

    def with_preset(self, preset: str) -> "SentenceLayout":
        """Another preset, the corpus's own element settings kept (seg[type=pw] stays wrap)."""
        return SentenceLayout(preset=preset, overrides=dict(self.overrides))

    # ── configuration ────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["SentenceLayout"]:
        """``{preset, elements: {name: {boundary, crossing}}}`` (application profile); None when absent."""
        if not data:
            return None
        if isinstance(data, str):
            return cls(preset=data)
        overrides = {}
        for name, spec in (data.get("elements") or {}).items():
            plain = name.split("[", 1)[0]
            preset_roles = PRESETS.get(data.get("preset") or DEFAULT_PRESET, {})
            base = preset_roles.get(name) or preset_roles.get(plain, UNKNOWN)
            overrides[name] = Role(spec.get("boundary", base.boundary), spec.get("crossing", base.crossing))
        return cls(preset=data.get("preset") or DEFAULT_PRESET, overrides=overrides)

    @classmethod
    def from_xml(cls, elem) -> Optional["SentenceLayout"]:
        """``<segmentation preset="…"><elements><item key="seg[type=pw]" name="seg" type="pw" boundary="…"
        crossing="…"/></elements></segmentation>`` — TEITOK's keyed list (xmlflatten makes it a hash; the key
        is unique: name, and [type=…] for elements of one @type); the older ``<element name type …/>``
        children are read too."""
        if elem is None:
            return None
        local = lambda e: (e.tag if isinstance(e.tag, str) else "").rsplit("}", 1)[-1]
        preset = (elem.get("preset") or DEFAULT_PRESET).strip()
        items = [c for c in elem if local(c) == "element"]
        for group in (c for c in elem if local(c) == "elements"):
            items += [c for c in group if local(c) == "item"]
        elements = {}
        for child in items:
            name = (child.get("name") or "").strip()
            if name:
                key = name + (f"[type={child.get('type').strip()}]" if child.get("type") else "")  # only seg with @type="pw"
            else:
                key = (child.get("key") or "").strip()  # an item with only a key: "seg" or "seg[type=pw]"
            if not key:
                continue
            elements[key] = {k: child.get(k).strip() for k in ("boundary", "crossing") if child.get(k)}
        return cls.from_dict({"preset": preset, "elements": elements})
