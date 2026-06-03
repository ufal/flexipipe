"""Parse ``--option namespace:key[=value]`` CLI extensions."""

from __future__ import annotations

from typing import Any


def _decode_option_value(val: str) -> str:
    """Decode common escapes in option values (e.g. ``\\n`` → newline)."""
    return (
        val.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
    )


def parse_flexipipe_options(option_strings: list[str]) -> dict[str, dict[str, Any]]:
    """Parse repeatable ``--option`` values into nested namespaces.

    Examples::

        tei-layout:normalize
        tei-layout:separator=\\n\\n
        tei-layout:block-tags=p,div,head
        punctuation-split:hard
        punctuation-split:boundary
        punctuation-split:harder
        punctuation-split:full
        punctuation-split:mode=harder
    """
    namespaces: dict[str, dict[str, Any]] = {}
    for raw in option_strings:
        item = (raw or "").strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"invalid --option {raw!r}: expected namespace:key or namespace:key=value"
            )
        ns, rest = item.split(":", 1)
        ns = ns.strip().lower()
        rest = rest.strip()
        if not ns or not rest:
            raise ValueError(
                f"invalid --option {raw!r}: expected namespace:key or namespace:key=value"
            )
        if "=" in rest:
            key, val = rest.split("=", 1)
            key = key.strip().lower()
            if not key:
                raise ValueError(f"invalid --option {raw!r}: empty key")
            namespaces.setdefault(ns, {})[key] = _decode_option_value(val)
        else:
            namespaces.setdefault(ns, {})[rest.lower()] = True
    return namespaces


def tei_layout_from_parsed_options(
    parsed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Map ``tei-layout`` namespace to xmltokenizer session layout kwargs."""
    tei = parsed.get("tei-layout", {})
    block_tags: tuple[str, ...] | None = None
    raw_tags = tei.get("block-tags")
    if raw_tags:
        if isinstance(raw_tags, str):
            block_tags = tuple(t.strip() for t in raw_tags.split(",") if t.strip())
        else:
            raise ValueError(
                "tei-layout:block-tags must be a comma-separated string"
            )
    separator = tei.get("separator", "\n")
    if not isinstance(separator, str) or not separator:
        raise ValueError("tei-layout:separator must be a non-empty string")
    normalize = tei.get("normalize", False)
    if normalize is True:
        layout_normalize = True
    elif normalize in (False, None):
        layout_normalize = False
    elif str(normalize).lower() in ("1", "true", "yes", "on"):
        layout_normalize = True
    elif str(normalize).lower() in ("0", "false", "no", "off"):
        layout_normalize = False
    else:
        raise ValueError(
            f"tei-layout:normalize must be a flag or boolean, got {normalize!r}"
        )
    return {
        "layout_normalize": layout_normalize,
        "layout_block_tags": block_tags,
        "layout_separator": separator,
    }
