"""``flexipipe release``: a corpus packed as a release for UD (a treebank) or UDMorph (UD annotation without
dependency relations), from any format flexipipe reads (CoNLL-U, TEITOK XML, folders of them).

    flexipipe release CORPUS... --lang xx --name NAME [--target auto|ud|udmorph] [--metadata meta.json]
                      [--previous splits.tsv|FOLDER] [-o OUT] [--zip] [--no-validate]

OUT/repo/        the files for the repository (data, README.md, LICENSE.txt): copy them into a clone and commit
OUT/splits.tsv   sent_id -> split: give it as --previous next time, so sentences keep their split
OUT/RELEASE.md   what was made, what the metadata still lack, the validation, the next steps
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ._cli_shared import add_logging_args, get_parent_parser

SMALL_SHARE = 0.1  # UD: below this share of the corpus with complete trees, UDMorph is suggested


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "release":
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        prog="flexipipe release",
        description="Pack a corpus as a UD treebank or a UDMorph dataset (files, splits, README, validation)",
        parents=[get_parent_parser()],
    )
    add_logging_args(parser)
    parser.add_argument("corpus", nargs="+", type=Path, help="CoNLL-U or TEITOK files, or folders with them")
    parser.add_argument("--lang", required=True, help="The language's code as UD has it (ISO 639-1 where there is one)")
    parser.add_argument("--name", required=True, help="UD: the treebank's name (UD_Language-NAME); UDMorph: the repository (xxx-variant-name)")
    parser.add_argument("--target", choices=["auto", "ud", "udmorph"], default="auto",
                        help="auto (default): UD when every word has a head and a relation, else UDMorph; "
                             "ud: documents without complete trees are left out")
    parser.add_argument("--metadata", type=Path, help="JSON with the README's data (language, summary, license, contributors …)")
    parser.add_argument("--previous", type=Path, help="The splits of an earlier release: its splits.tsv, or a folder with its CoNLL-U files")
    parser.add_argument("--xpos-attr", help="For TEITOK files: the attribute with the XPOS (default: as the corpus's settings say, else xpos)")
    parser.add_argument("-o", "--output", type=Path, help="The package folder (default: release-NAME)")
    parser.add_argument("--zip", action="store_true", help="Also write the package as OUT.zip")
    parser.add_argument("--no-validate", action="store_true", help="Do not run the UD validator")
    parser.add_argument("--json", action="store_true", help="The outcome as JSON on stdout (for TEITOK's release page)")
    args = parser.parse_args(argv)

    from . import release as R
    from .release_readme import license_text, ud_readme, udmorph_readme

    meta = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
    settings = _teitok_settings(args.corpus)
    xpos_attr = args.xpos_attr or (settings.resolve_xml_attribute("xpos", "xpos") if settings else "xpos")
    comments = (settings.sentence_comments or None) if settings else None
    docs = R.load(args.corpus, xpos_attr=xpos_attr, sentence_comments=comments)
    words = sum(d.words for d in docs)
    if not words:
        return _fail(args, "No words found in the corpus given")
    cov = R.coverage(docs)
    target = R.choose_target(cov) if args.target == "auto" else args.target
    notes = []
    left_out = []
    if target == "ud":
        # UD takes complete trees only: documents with words without a head or relation stay out
        docs, left_out = R.complete_docs(docs)
        kept = sum(d.words for d in docs)
        if not kept:
            return _fail(args, "No document has a head and a relation for every word, so there is nothing for UD: "
                               "release it for UDMorph instead (--target udmorph)")
        if left_out:
            if kept / words < SMALL_SHARE:
                notes.append(f"Only {kept / words:.0%} of the corpus ({kept} of {words} words) has complete trees: "
                             "a UDMorph release (--target udmorph) of the whole corpus may serve it better")
            notes.append(f"Left out, as not every word has a head and a relation: {len(left_out)} documents, "
                         f"{words - kept} words ({_names(left_out)})")
        words = kept
        cov = R.coverage(docs)
    if cov["upos"] < 0.999:
        notes.append(f"Only {cov['upos']:.0%} of the words have a UPOS")

    want = R.targets(words, target)
    previous = R.read_splits(args.previous) if args.previous else {}
    gone = sum(1 for d in left_out for x in d.sents if x.sent_id in previous)
    if gone:
        notes.append(f"{gone} sentences of the earlier release are left out now (their documents no longer have complete trees)")
    split_of = R.assign(R.units(docs, want), want, previous)
    texts = R.split_texts(docs, split_of, target=target)
    stats = {sp: {"sents": t[1], "words": t[2]} for sp, t in texts.items()}

    name = args.name
    if target == "ud":
        language = meta.get("language") or args.lang
        repo_name = f"UD_{language.replace(' ', '_')}-{name}"
        stem = f"{args.lang}_{name.lower()}-ud"
        readme, problems = ud_readme(meta, stats, cov)
    else:
        repo_name = name
        stem = name
        readme, problems = udmorph_readme(meta, stats, cov, name)
    lic, lic_problems = license_text(meta)
    problems += lic_problems

    out = args.output or Path(f"release-{name}")
    repo = out / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    for sp, (text, _, _) in texts.items():
        (repo / f"{stem}-{sp}.conllu").write_text(text, encoding="utf-8")
    (repo / "README.md").write_text(readme, encoding="utf-8")
    if lic:
        (repo / "LICENSE.txt").write_text(lic, encoding="utf-8")
    (out / "splits.tsv").write_text("".join(f"{sid}\t{sp}\n" for sid, sp in split_of.items()), encoding="utf-8")
    moved = sum(1 for sid, sp in split_of.items() if sid in previous and previous[sid] != sp)
    if moved:
        notes.append(f"{moved} sentences changed split since the earlier release")

    validation = None if args.no_validate else _validate(repo, args.lang, out / "validation.txt")
    report = _report(target, repo_name, stats, cov, notes, problems, validation, len(previous))
    (out / "RELEASE.md").write_text(report, encoding="utf-8")
    zipfile = None
    if args.zip:
        zipfile = shutil.make_archive(str(out), "zip", root_dir=out)

    outcome = {"target": target, "repository": repo_name, "folder": str(out), "zip": zipfile, "splits": stats,
               "coverage": cov, "left_out": [d.source or d.id for d in left_out], "notes": notes, "metadata_problems": problems, "validation": validation}
    if args.json:
        print(json.dumps(outcome, ensure_ascii=False))
    else:
        print(f"[flexipipe] {target.upper()} release {repo_name}: " + ", ".join(f"{sp} {s['words']} words" for sp, s in stats.items()))
        for n in notes + problems:
            print(f"[flexipipe]   {n}")
        if validation:
            print(f"[flexipipe] Validation: {validation['summary']}")
        print(f"[flexipipe] Written: {out}" + (f" and {zipfile}" if zipfile else "") + " (see RELEASE.md)")
    return 0


def _names(docs, most: int = 10) -> str:
    """The files (or documents) left out, the first few by name."""
    names = list(dict.fromkeys(d.source or d.id for d in docs))
    return ", ".join(names[:most]) + (f" and {len(names) - most} more" if len(names) > most else "")


def _teitok_settings(paths):
    """For TEITOK input (XML files, or a project folder): the corpus's settings (attribute names, sentence comments)."""
    from .release import _files

    if not any(f.suffix == ".xml" for f in _files(paths)):
        return None
    try:
        from .teitok_settings import find_settings_xml, load_teitok_settings
    except Exception:
        return None
    for p in paths:
        found = find_settings_xml(p if p.is_file() else p / "xmlfiles" / "x.xml")
        if found:
            return load_teitok_settings(settings_path=found)
    return None


def _validate(repo: Path, lang: str, report: Path) -> dict:
    """The UD validator (flexipipe validate: missing annotation layers are not counted as errors)."""
    files = [str(f) for f in sorted(repo.glob("*.conllu"))]  # the validator reads files, not a folder
    cmd = [sys.executable, "-m", "flexipipe", "validate", *files, "--validator-lang", lang]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception as e:
        return {"ok": False, "summary": f"the validator could not be run ({e})"}
    text = (r.stdout or "") + (r.stderr or "")
    report.write_text(text, encoding="utf-8")
    if "depends on the 'udapi' package" in text:
        return {"ok": False, "summary": "not run: the UD validator needs udapi (pip install udapi)", "file": str(report)}
    missing = re.search(r"ModuleNotFoundError: No module named '([^']+)'", text)
    if missing:  # this Python lacks one of flexipipe's own packages
        return {"ok": False, "summary": f"not run: this Python lacks the package {missing.group(1)}", "file": str(report)}
    last = [l for l in text.splitlines() if l.strip()][-3:]
    return {"ok": r.returncode == 0, "summary": ("passed" if r.returncode == 0 else "errors, see validation.txt") + (": " + " / ".join(last) if last else ""),
            "file": str(report)}


def _report(target, repo_name, stats, cov, notes, problems, validation, previous) -> str:
    L = [f"# {'UD treebank' if target == 'ud' else 'UDMorph dataset'}: {repo_name}", ""]
    L += ["| split | sentences | words |", "|---|---|---|"]
    L += [f"| {sp} | {s['sents']} | {s['words']} |" for sp, s in stats.items()]
    L += ["", "Annotated: " + ", ".join(f"{f} {v:.0%}" for f, v in cov.items()), ""]
    if previous:
        L += [f"Splits kept from the earlier release for {previous} sentences.", ""]
    if notes:
        L += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    if problems:
        L += ["## README / LICENSE: still to fill in", ""] + [f"- {p}" for p in problems] + [""]
    if validation:
        L += ["## Validation", "", validation["summary"], ""]
    L += ["## Next steps", ""]
    if target == "ud":
        L += [f"1. A new treebank: ask the UD release team for the repository {repo_name} "
              "(https://universaldependencies.org/contributing/); an existing one: clone it.",
              f"2. `git checkout dev` (UD's maintainers never touch master), copy the files of `repo/` in, and look at `git diff`.",
              "3. Commit and push to dev; the next release validates and picks it up.",
              "4. Keep `splits.tsv`: give it as --previous next time, so sentences keep their split."]
    else:
        L += ["1. A new dataset: open an issue at https://github.com/UDMorph/udmorph_contributions with this package; "
              f"an existing one: clone https://github.com/UDMorph/{repo_name}.",
              "2. Copy the files of `repo/` in, look at `git diff`, commit and push (or send a pull request).",
              "3. Keep `splits.tsv`: give it as --previous next time, so sentences keep their split."]
    return "\n".join(L) + "\n"


def _fail(args, msg: str) -> int:
    if args.json:
        print(json.dumps({"error": msg}))
    else:
        print(f"[flexipipe] {msg}", file=sys.stderr)
    return 1
