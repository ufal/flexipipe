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

## Evaluation and debugging

* **`--verbose`**: High-level progress and evaluation summaries.
* For troubleshooting, **`--debug`** (on `process` and `benchmark`) enables: curl-style REST payloads, tokenization diff samples, backend logs. Not shown in main help; for developer use.
* `tmp/` directories can hold per-run artifacts for inspection.

---

↑ [Documentation index](README.md) · [Home](Home.md) · [Installation](Installation.md) · [Quick Start](Quick-Start.md) · [Backends](Backends.md) · [Contributing](Contributing.md)
