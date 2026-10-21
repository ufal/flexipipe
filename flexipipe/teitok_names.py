"""Write back under the names the corpus uses. TEITOK corpora name the same layer differently:
the normalized form @reg / @nform / @norm, the expansion @expan / @fform, the tag @xpos / @tag / @pos /
@msd, a pronunciation @pron / @ipa / @sampa. When the document — or, for one without tokens yet, a few
other documents of the corpus — already uses one of a layer's names and the settings' choice is not
among them, that name comes first in the settings' mapping, so the run writes (and reads) it there.

The user decides where it matters: settings.xml <flexipipe><layers><item key="xpos" attribute="xpos"/></layers> (the
TEITOK set-up page: "Where flexipipe writes"), a pattribute's flexipipe="…", or --attrs-map — those
layers are left alone.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# internal layer -> the names corpora give it (most usual first)
NAMES: Dict[str, List[str]] = {
    "xpos": ["xpos", "tag", "pos", "msd"],
    "reg": ["reg", "nform", "norm"],
    "expan": ["expan", "fform"],
    "trslit": ["trslit", "translit", "roman"],
    "lemma": ["lemma"],
    # the canonical pronunciation (task pron); not @phon, the realised one (dev/SPOKEN-PRONUNCIATION-AND-SENTENCES.md)
    "pron": ["pron", "ipa", "sampa", "xsampa"],
}
SAMPLE = 5  # other documents read when the document itself has no tokens yet


def _tok_attributes(path: Path) -> Counter:
    from lxml import etree

    used: Counter = Counter()
    try:
        parser = etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True)
        for el in etree.parse(str(path), parser).iter():
            tag = el.tag if isinstance(el.tag, str) else ""
            if tag.rsplit("}", 1)[-1] in ("tok", "dtok"):
                used.update(k for k, v in el.attrib.items() if v.strip() not in ("", "_"))
    except (OSError, etree.XMLSyntaxError):
        pass
    return used


def adopt_corpus_names(settings, document: Optional[Path], corpus_root: Optional[Path] = None) -> Dict[str, str]:
    """Put the name the corpus uses first in the settings' mapping, per layer: the name the document
    uses, else (the layer not in the document yet) the one a few other documents use. Returns
    {layer: name} for the layers changed."""
    if settings is None:
        return {}
    in_doc = _tok_attributes(Path(document)) if document and Path(document).is_file() else Counter()
    in_corpus: Optional[Counter] = None
    # set by the user: settings.xml <flexipipe><layers><item key attribute> or a pattribute's flexipipe="…"
    explicit = (set(getattr(settings, "explicit_flexipipe_mappings", {}).values()) | set(getattr(settings, "layer_attributes", {}))
                | set(getattr(settings, "excluded_layers", ())))
    changed: Dict[str, str] = {}
    for layer, names in NAMES.items():
        if layer in explicit:
            continue
        used = in_doc
        if not any(n in in_doc for n in names) and corpus_root is not None:
            if in_corpus is None:
                in_corpus = Counter()
                for other in sorted((Path(corpus_root) / "xmlfiles").rglob("*.xml"))[:SAMPLE]:
                    in_corpus.update(_tok_attributes(other))
            used = in_corpus
        current = settings.resolve_xml_attribute(layer, default=layer)
        present = [n for n in names if n in used]
        if not present or current in present:
            continue
        name = max(present, key=lambda n: used[n])
        settings.attribute_mappings[layer] = [name] + [n for n in settings.attribute_mappings.get(layer, []) if n != name]
        changed[layer] = name
    return changed


def automatic_choices(settings, corpus_root: Path) -> Dict[str, dict]:
    """Per layer: what the automatic choice writes to in this corpus (as for a document not tokenized
    yet) and the names found — for the set-up page. Leaves `settings` as it was."""
    import copy

    probe = copy.deepcopy(settings)
    probe.layer_attributes = {}
    probe._build_attribute_mappings()  # the mapping without the user's choices
    adopt_corpus_names(probe, None, corpus_root)
    used: Counter = Counter()
    for other in sorted((Path(corpus_root) / "xmlfiles").rglob("*.xml"))[:SAMPLE]:
        used.update(_tok_attributes(other))
    return {layer: {"automatic": probe.resolve_xml_attribute(layer, default=layer),
                    "set": "-" if layer in getattr(settings, "excluded_layers", ()) else settings.layer_attributes.get(layer),
                    "names": NAMES[layer],
                    "in_corpus": [n for n, _ in used.most_common()]}
            for layer in NAMES}
