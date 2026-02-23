# flexiPipe

flexiPipe is a modular NLP pipeline for Universal Dependencies data that glues rule-based components, the legacy flexitag tagger, and multiple neural backends (SpaCy, Stanza, Flair, UDPipe REST, UDMorph REST, and UD-Kanbun). It can ingest raw text, CoNLL-U, and TEITOK XML, preserves existing annotations, and exports CoNLL-U (with optional implicit MWTs and IOB NER), TEITOK with nested `<tok>`/`<dtok>`/`<name>` structures, SVG dependency tree visualizations, HTML interactive trees, and LaTeX/TikZ dependency trees.

If you have installed the wrapper, you can run `flexipipe` instead of `python -m flexipipe`.

## Commands

| Command | Purpose |
| --- | --- |
| `flexipipe process` | Tag/parse/normalize input with any backend |
| `flexipipe benchmark` | Evaluate backends against gold treebanks |
| `flexipipe train` | Train flexitag or (where implemented) neural backends |
| `flexipipe convert` | Convert between formats (tagged files, treebanks, lexicons) |
| `flexipipe info` | List backends, models, examples, tasks, languages, renderers, or installation details |
| `flexipipe config` | Inspect or change defaults (models dir, backend, output format, language detector, implicit MWT) |
| `flexipipe install` | Install optional backends, install the launcher (`flexipipe install wrapper`), or upgrade flexipipe (`flexipipe install update`) |

Use `flexipipe info backends`, `flexipipe info models --backend <name>`, etc. to explore. Get detailed help: **`flexipipe <subcommand> --help`** (e.g. `flexipipe config --help`, `flexipipe process --help`).

---

## Key features

* **Multi-backend orchestration**: flexitag, spacy, stanza, flair, transformers, udpipe, udmorph, udkanbun, treetagger, nametag, and more. Use `flexipipe info backends` to see available backends; more can be installed via `flexipipe install <name>`.
* **Input flexibility**: auto-detects format or `--input-format` (`auto`, `conllu`, `tei`, `raw`). Raw mode can read from STDIN.
* **Backend chaining**: Pipe output from one backend to another (e.g. UDPipe then NameTag for NER).
* **Document representation**: multi-word tokens, tokids, `space_after`, named entities (CoNLL-U `Entity=B-ORG`, TEITOK `<name>`), confidence slots.
* **Output control**: CoNLL-U, TEITOK, SVG/HTML/LaTeX dependency trees, JSON. Configure defaults via `flexipipe config`.
* **Evaluation**: benchmark backends on gold treebanks; alignment via tokids and SequenceMatcher.
* **Centralised model storage**: `~/.flexipipe/models/<backend>/` (overridable via config or `FLEXIPIPE_MODELS_DIR`).
* **Rich configuration**: `config.json` for default backend, output format, implicit MWT, language detector; `flexipipe config --show` to inspect.

---

## Wiki pages

* [Installation](Installation) — requirements, pip install, wrapper, optional extras, language detection
* [Quick Start](QuickStart.md) — tagging, benchmarking, training, configuration examples
* [Backends](Backends) — backend overview table, Transformers backend details
* [Reference](Reference) — input/output formats, NER, MWT, info commands, convert, evaluation tips
* [Contributing](Contributing) — project layout, development notes
