# Reference

## Input formats

* **CoNLL-U**: `--input-format conllu` or auto-detected via `.conllu`.
* **TEITOK XML**: `--input-format tei`. Converts `<tok>`/`<dtok>` to internal representation (preserves TokId).
* **Raw / STDIN**: `--input-format raw` or `--input -`. Flexitag segments/tokenizes; neural backends receive raw text.
* **Tokenized CoNLL-U**: benchmark/check can use `tokenized` mode, merging UD MWTs as needed.

## Output formats

* **CoNLL-U**: Entity= entries, optional TokId, implicit MWT ranges.
* **CoNLL-U with NER**: `--output-format conllu-ne`.
* **TEITOK**: `<dtok/>`, `<name>` for entities, `xml:id` for TokIds.
* **SVG**: Dependency tree (styles `dep`, `tree`; options like `tree,boxes`, `tree,root`). Renderers: displacy, udapi, conllview, graphviz. `flexipipe info renderers` to list.
* **HTML / LaTeX**: Interactive or TikZ dependency trees (requires `flexipipe install udapi`).
* **JSON**: Full document representation.

## Named entity recognition

* NER-capable backends set `Sentence.entities`. CoNLL-U: `Entity=B-ORG`, etc. TEITOK: `<name type="ORG">` around tokens.

## Multi-word tokens (MWTs)

* Preserved via tokid alignment. `--create-implicit-mwt` (or config default) can add MWT ranges for contractions. TEITOK exporter keeps `<tok>` text when `<dtok>` children exist.

## Model and data management

* Models: `~/.flexipipe/models/<backend>/`. Override with `FLEXIPIPE_MODELS_DIR` or `flexipipe config --set-models-dir`. Backend envs (e.g. Stanza) point to this directory. Config: `~/.flexipipe/config.json`; edit via CLI.

## Information commands

```bash
flexipipe info backends
flexipipe info models --backend transformers
flexipipe info examples
flexipipe info tasks
flexipipe info languages
flexipipe info renderers
flexipipe info installation   # version, package path, config/models dirs
```

## Format conversion

```bash
# TEITOK or CoNLL-U to train/dev/test splits
flexipipe convert --type treebank --input /path/to/corpus --output /path/to/output --train-ratio 0.8 --dev-ratio 0.1 --test-ratio 0.1
```

Rerunning preserves splits by `sent_id`. Handles missing sections.

## A release for UD or UDMorph

A corpus (CoNLL-U or TEITOK files, or folders of them) packed as a UD treebank, or as a
[UDMorph](https://lindat.mff.cuni.cz/services/teitok-live/udmorph/) dataset when it has UD annotation without
(complete) dependency relations:

```bash
flexipipe release /path/to/corpus --lang osp --name ISWOC --metadata meta.json --zip
flexipipe release /path/to/corpus --lang osp --name ISWOC --previous release-ISWOC/splits.tsv   # the next release
```

The package holds the repository's files (`repo/`: train/dev/test under UD's or UDMorph's names, README.md with UD's
metadata block or UDMorph's field table, LICENSE.txt), `splits.tsv`, the validator's report and `RELEASE.md`. That last
file lists what the README still lacks and the next steps. The splits follow UD's sizes, on document boundaries where
documents are small enough. With `--previous`, sentences keep their split. For UD (`--target ud`), documents in which not every word
has a head and a relation are left out and listed in the report; when that leaves nothing, or less than a tenth of
the corpus, it suggests a UDMorph release instead. The metadata JSON takes the README's fields:
`language`, `summary`, `introduction`, `license`, `contributors`, `contact`, `genre`, `lemmas`, `upos`, `xpos`,
`features`, `relations`, `since`, `contributing`, `citation`, `acknowledgments`, and for UDMorph `variant`, `region`,
`title`, `source`.

## A language's UD inventory

For a language UD has no data for yet: the relations, UPOS and features (values per UPOS) a corpus uses, as a draft
to complete. Optionally on top of a related language's UD data; what the corpus uses that this language does not
permit is listed, to decide on.

```bash
flexipipe udlang --lang osp --base es /path/to/corpus        # CoNLL-U or TEITOK files, or folders
flexipipe udlang --lang es --ud-only                         # UD's own data for a language, in short
```

The file (`deptree-edit-ud-LANG.json`) is the one TEITOK's tree editor (teitok-extras) reads from `Resources/`;
its language-data page completes it.

## Evaluation and debugging

* **`--verbose`**: High-level progress and evaluation summaries.
* For troubleshooting, **`--debug`** (on `process` and `benchmark`) enables: curl-style REST payloads, tokenization diff samples, backend logs. Not shown in main help; for developer use.
* `tmp/` directories can hold per-run artifacts for inspection.

---

↑ [Documentation index](README.md) · [Home](Home.md) · [Installation](Installation.md) · [Quick Start](Quick-Start.md) · [Backends](Backends.md) · [Contributing](Contributing.md)
