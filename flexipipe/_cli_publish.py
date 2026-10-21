"""``flexipipe publish``: a trained model made ready to deposit (flexipipe-hub, or anywhere else).

    flexipipe publish --check FOLDER             what the folder gives away about this machine
    flexipipe publish --pack FOLDER [-o FILE]    a .tar.gz without it, to deposit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._cli_shared import add_logging_args, get_parent_parser


def _show(title: str, findings) -> None:
    if not findings:
        return
    print(title)
    for f, where, value in findings:
        print(f"  {f}  {where}: {value}")


def _corpus_licence(folder: Path) -> None:
    try:
        manifest = json.loads((folder / "flexipipe-model.json").read_text(encoding="utf-8"))
    except Exception:
        return
    from .corpus_licence import describe

    if "corpus_licence" in manifest:
        print(f"[flexipipe] What the training files say about their licence: {describe(manifest['corpus_licence'])}")
        print("[flexipipe] The model holds material from those texts: publish it only as far as that licence allows.")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "publish":
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        prog="flexipipe publish",
        description="Make a trained model ready to publish: without the paths of this machine",
        parents=[get_parent_parser()],
    )
    add_logging_args(parser)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--check", metavar="FOLDER", type=Path, help="Show what the model folder records about this machine")
    what.add_argument("--pack", metavar="FOLDER", type=Path, help="Write a .tar.gz of the model folder without it")
    parser.add_argument("-o", "--output", type=Path, help="For --pack: the package to write (default: FOLDER.tar.gz next to the folder)")
    parser.add_argument("--json", action="store_true",
                        help="For --pack: the outcome as JSON, with the deposit form's fields (for TEITOK's publish page)")
    args = parser.parse_args(argv)

    from .model_pack import check, pack

    if args.check:
        if not args.check.is_dir():
            print(f"[flexipipe] {args.check} is not a model folder")
            return 1
        cleanable, stuck = check(args.check)
        if not cleanable and not stuck:
            print(f"[flexipipe] {args.check}: no paths of this machine")
        _show("[flexipipe] Paths in the metadata, which --pack removes (each becomes its last part):", cleanable)
        _show("[flexipipe] Paths in the model itself — --pack leaves them (they may be data); look at them:", stuck)
        _corpus_licence(args.check)
        return 0

    from .deposit_card import corpus_tagset, deposit_fields

    tagset = corpus_tagset(args.pack) if args.pack.is_dir() else None  # a corpus's own tagset goes along
    try:
        package, cleaned, stuck = pack(args.pack, args.output, {"tagset.xml": tagset} if tagset else None)
    except (ValueError, RuntimeError) as e:
        if args.json:
            print(json.dumps({"ok": False, "error": str(e)}))
        else:
            print(f"[flexipipe] {e}")
        return 1
    if args.json:
        from .corpus_licence import describe
        from .model_hub import hub_url

        try:
            manifest = json.loads((args.pack / "flexipipe-model.json").read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        rec = manifest.get("corpus_licence")
        print(json.dumps({
            "ok": True, "package": str(package), "size": package.stat().st_size,
            "cleaned": [list(x) for x in cleaned], "look": [list(x) for x in stuck],
            "tagset": bool(tagset), "corpus_licence": rec, "corpus_licence_text": describe(rec) if rec else "",
            "fields": deposit_fields(args.pack, tagset_included=bool(tagset)), "hub_url": hub_url(),
        }, ensure_ascii=False))
        return 0
    _show("[flexipipe] Removed from the metadata (in the package; the folder keeps them):", cleaned)
    _show("[flexipipe] Left in the model itself (they may be data) — look at them before depositing:", stuck)
    if tagset:
        print(f"[flexipipe] The corpus's tagset description is in the package (tagset.xml)")
    print(f"[flexipipe] ✓ {package} ({package.stat().st_size / 1048576:.1f} MB): ready to deposit")
    _corpus_licence(args.pack)
    return 0
