# flexiPipe

flexiPipe is a modular NLP pipeline for Universal Dependencies data: rule-based components, the flexitag tagger, and multiple neural backends (SpaCy, Stanza, Flair, UDPipe, UDMorph, UD-Kanbun, etc.). It ingests raw text, CoNLL-U, and TEITOK XML and exports CoNLL-U, TEITOK, SVG/HTML/LaTeX trees, and more.

**Full documentation:** see the [wiki/](wiki/) folder — [wiki/README.md](wiki/README.md) is the index; installation, backends, quick start, and reference are in the linked pages.

---

## Install

```bash
pip install flexipipe
# Optional backends (use flexipipe info backends to see all):
flexipipe install spacy
flexipipe install udapi   # for HTML/LaTeX output
```

Then run `flexipipe config --wizard` to pick defaults. To run as `flexipipe` instead of `python -m flexipipe`, install the wrapper: `python -m flexipipe install wrapper` (see [Installation](wiki/Installation.md)).

---

## Quick start

```bash
# Tag text (default backend)
echo "Don't even think he wouldn't do it." | flexipipe process --input -

# With a backend and model
flexipipe process --backend spacy --model en_core_web_sm --input file.txt --output out.conllu

# See available backends and subcommand help
flexipipe info backends
flexipipe process --help
```

---

## Commands

| Command | Purpose |
| --- | --- |
| `flexipipe process` | Tag/parse/normalize input |
| `flexipipe benchmark` | Evaluate backends on gold data |
| `flexipipe train` | Train flexitag or backend models |
| `flexipipe convert` | Convert formats (e.g. TEITOK → CoNLL-U) |
| `flexipipe info` | List backends, models, examples, tasks, languages, installation |
| `flexipipe config` | Set defaults (backend, output format, models dir) |
| `flexipipe install` | Install optional backends or the launcher; upgrade flexipipe |

Use **`flexipipe <subcommand> --help`** for full options. Use **`flexipipe info backends`** to see available backends (more can be installed via `flexipipe install <name>`).

---

## Project layout

- `flexipipe/` — main package (CLI, backends, converters)
- `flexitag/` — C++ flexitag sources and bindings
- `README_CPP.md` — native build instructions
- `wiki/` — full documentation (Markdown); [wiki/README.md](wiki/README.md) is the index.

See [Contributing](wiki/Contributing.md) for development notes.
