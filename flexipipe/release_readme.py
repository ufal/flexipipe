"""The README and LICENSE of a release package: UD's README.md (sections, and the machine-readable metadata block at
the end, as https://universaldependencies.org/contributing/repository_files.html asks) or a UDMorph README (the field
table of the UDMorph repositories). The metadata come from a JSON file; what is missing is listed, not invented."""
from __future__ import annotations

from typing import Dict, List, Tuple

LICENSES = {
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-NC 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC BY-ND 4.0": "https://creativecommons.org/licenses/by-nd/4.0/",
    "CC BY-NC-ND 4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "CC0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}

# UD's metadata fields: (README label, metadata key, allowed values or None)
UD_FIELDS = [
    ("Data available since", "since", None),
    ("License", "license", None),
    ("Includes text", "includes_text", ("yes", "no")),
    ("Genre", "genre", None),
    ("Lemmas", "lemmas", ("manual native", "converted from manual", "converted with corrections", "automatic", "automatic with corrections", "not available")),
    ("UPOS", "upos", ("manual native", "converted from manual", "converted with corrections")),
    ("XPOS", "xpos", ("manual native", "automatic", "automatic with corrections", "not available")),
    ("Features", "features", ("manual native", "converted from manual", "converted with corrections", "automatic", "automatic with corrections", "not available")),
    ("Relations", "relations", ("manual native", "converted from manual", "converted with corrections")),
    ("Contributors", "contributors", None),
    ("Contributing", "contributing", ("here", "here source", "elsewhere", "to be adopted")),
    ("Contact", "contact", None),
]


def _missing(meta: Dict, keys: List[str]) -> List[str]:
    return [k for k in keys if not str(meta.get(k, "")).strip()]


def ud_readme(meta: Dict, stats: Dict, cov: Dict[str, float]) -> Tuple[str, List[str]]:
    """UD's README.md, and what the metadata lack (or have outside UD's allowed values)."""
    meta = dict(meta)
    # what the data say for themselves
    meta.setdefault("includes_text", "yes")
    meta.setdefault("parallel", "no")
    if cov.get("lemma", 0) == 0:
        meta.setdefault("lemmas", "not available")
    if cov.get("xpos", 0) == 0:
        meta.setdefault("xpos", "not available")
    if cov.get("feats", 0) == 0:
        meta.setdefault("features", "not available")
    problems = [f"{k} (missing)" for k in _missing(meta, ["summary", "introduction"] + [k for _, k, _ in UD_FIELDS])]
    for label, k, allowed in UD_FIELDS:
        v = str(meta.get(k, "")).strip()
        if v and allowed and v not in allowed:
            problems.append(f"{k}: '{v}' is not one of: {', '.join(allowed)}")
    L = ["# Summary", "", meta.get("summary", ""), "", "# Introduction", "", meta.get("introduction", ""), ""]
    L += ["## Data split", "", _split_text(stats), ""]
    if meta.get("acknowledgments") or meta.get("citation"):
        L += ["# Acknowledgments", "", meta.get("acknowledgments", "")]
        if meta.get("citation"):
            L += ["", "## References", "", meta["citation"]]
        L += [""]
    if meta.get("changelog"):
        L += ["# Changelog", "", meta["changelog"], ""]
    L += ["=== Machine-readable metadata (DO NOT REMOVE!) ==="]
    for label, k, _ in UD_FIELDS[:3]:
        L.append(f"{label}: {meta.get(k, '')}")
    L.append(f"Parallel: {meta.get('parallel', 'no')}")
    for label, k, _ in UD_FIELDS[3:]:
        L.append(f"{label}: {meta.get(k, '')}")
    L.append("===============================================================================")
    return "\n".join(L) + "\n", problems


def udmorph_readme(meta: Dict, stats: Dict, cov: Dict[str, float], name: str) -> Tuple[str, List[str]]:
    """A UDMorph README: the field table of the UDMorph repositories, the description, license and citation."""
    ann = [f for f, k in (("lemma", "lemma"), ("upos", "upos"), ("xpos", "xpos"), ("feats", "feats")) if cov.get(k, 0) > 0]
    title = meta.get("language", "") + (f" ({meta['variant']})" if meta.get("variant") else "") + (f" - {meta['title']}" if meta.get("title") else "")
    problems = [f"{k} (missing)" for k in _missing(meta, ["language", "license", "contributors"])]
    rows = [("Language:", meta.get("language", "")), ("Region:", meta.get("region", "")), ("Annotations:", ", ".join(ann)),
            ("Source:", meta.get("source", "")), ("Contributors:", meta.get("contributors", "")),
            ("UDMorph:", f"[{name}](https://lindat.mff.cuni.cz/services/teitok-live/udmorph/{name}/)")]
    L = [f"# {title.strip() or name}", "", "| Field | Value |", "| ---- | ---- |"]
    L += [f"| {k} | {v} |" for k, v in rows if v]
    L += [""]
    if meta.get("summary") or meta.get("introduction"):
        L += ["## Description", "", meta.get("summary", ""), "", meta.get("introduction", ""), ""]
    L += ["## Data split", "", _split_text(stats), ""]
    L += ["## License and Citation", "", f"The corpus is available under the license {meta.get('license', '')}.", ""]
    if meta.get("acknowledgments"):
        L += [meta["acknowledgments"], ""]
    if meta.get("citation"):
        L += ["If you use this corpus, please cite:", "", meta["citation"], ""]
    return "\n".join(L).rstrip() + "\n", problems


def _split_text(stats: Dict) -> str:
    parts = [f"{sp}: {stats[sp]['sents']} sentences, {stats[sp]['words']} words" for sp in ("train", "dev", "test") if sp in stats]
    return "; ".join(parts) + ". Split on document boundaries where documents are small enough, else in runs of" \
        " consecutive sentences; sentences keep their split from one release to the next."


def license_text(meta: Dict) -> Tuple[str, List[str]]:
    lic = str(meta.get("license", "")).strip()
    if not lic:
        return "", ["license (missing): no LICENSE.txt"]
    url = meta.get("license_url") or LICENSES.get(lic, "")
    holder = meta.get("rights") or meta.get("contributors", "")
    text = f"This work is licensed under the {lic} license" + (f" ({url})" if url else "") + ".\n"
    if holder:
        text += f"\nCopyright: {holder}\n"
    return text, ([] if url else [f"license '{lic}': not a known licence name, give license_url"])
