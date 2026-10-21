"""``flexipipe udlang``: a language's UD inventory drafted from a corpus, for a language UD has no data for yet.

    flexipipe udlang --lang xx CORPUS... [--base es] [-o FILE]
    flexipipe udlang --lang es --ud-only [-o FILE]       UD's own data for a language, in short

The file is TEITOK's ``Resources/deptree-edit-ud-LANG.json`` (teitok-extras' tree editor and its language-data page).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._cli_shared import add_logging_args, get_parent_parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "udlang":
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        prog="flexipipe udlang",
        description="Draft a language's UD inventory (relations, UPOS, features per UPOS) from a corpus",
        parents=[get_parent_parser()],
    )
    add_logging_args(parser)
    parser.add_argument("corpus", nargs="*", type=Path, help="CoNLL-U or TEITOK files, or folders with them")
    parser.add_argument("--lang", required=True, help="The language's code (UD's: ISO 639-1 where there is one)")
    parser.add_argument("--base", help="A related language whose UD data the draft starts from (e.g. es)")
    parser.add_argument("--ud-only", action="store_true", help="Only UD's own data for --lang, no corpus")
    parser.add_argument("--min", type=int, default=1, dest="minimum",
                        help="Leave out what the corpus uses fewer times than this (default 1: everything)")
    parser.add_argument("--xpos-attr", default="xpos", help="For TEITOK files: the attribute with the XPOS")
    parser.add_argument("--no-ud", action="store_true", help="Do not fetch UD's data (no base, nothing marked as new)")
    parser.add_argument("-o", "--output", type=Path, help="The file to write (default: deptree-edit-ud-LANG.json)")
    args = parser.parse_args(argv)

    from .model_storage import get_cache_dir
    from .udlang import compact, count, draft, load_ud_data, universal

    if not args.corpus and not args.ud_only:
        parser.error("give the corpus files or folders (or --ud-only)")
    ud = base = None
    if not args.no_ud:
        try:
            feats_json, deprels_json = load_ud_data(get_cache_dir() / "ud-data")
        except Exception as e:
            if args.base or args.ud_only:
                print(f"[flexipipe] UD's data could not be fetched ({e}); --no-ud drafts from the corpus alone", file=sys.stderr)
                return 1
            print(f"[flexipipe] UD's data could not be fetched ({e}): the draft is the corpus's alone", file=sys.stderr)
        else:
            ud = universal(feats_json, deprels_json)
            if args.ud_only:
                out = compact(feats_json, deprels_json, args.lang)
                if not out:
                    print(f"[flexipipe] UD has no data for {args.lang}", file=sys.stderr)
                    return 1
                return _write(out, args)
            if args.base:
                base = compact(feats_json, deprels_json, args.base)
                if not base:
                    print(f"[flexipipe] UD has no data for {args.base}", file=sys.stderr)
                    return 1
    if args.ud_only:
        parser.error("--ud-only needs UD's data (not with --no-ud)")

    counts = count(args.corpus, xpos_attr=args.xpos_attr)
    if not counts["words"]:
        print("[flexipipe] No words found in the corpus given", file=sys.stderr)
        return 1
    out = draft(counts, args.lang, base=base, ud=ud, minimum=args.minimum)
    d = out["draft"]
    print(f"[flexipipe] {d['files']} files, {d['words']} words: {len(out['deprels'])} relations, "
          f"{len(out['upos'])} UPOS, {len(out['feats'])} features" + (f" (on top of {args.base})" if base else ""))
    if d["new"]["deprels"] or d["new"]["feats"]:
        print(f"[flexipipe] Not in UD: {', '.join(d['new']['deprels'] + d['new']['feats'])}")
    beyond = d["beyond_base"]["deprels"] + d["beyond_base"]["feats"]
    if beyond:
        print(f"[flexipipe] Used, but not permitted in {args.base}: {', '.join(beyond)}")
    return _write(out, args)


def _write(out: dict, args) -> int:
    path = args.output or Path(f"deptree-edit-ud-{out['lang']}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[flexipipe] Written: {path}")
    return 0
