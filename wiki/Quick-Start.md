# Quick Start

## Tagging

```bash
# Raw text via STDIN, default backend
echo "Don't even think he wouldn't do it." | flexipipe process --input -

# CoNLL-U file with SpaCy, export TEITOK XML
flexipipe process --input data/en.conllu --backend spacy --model en_core_web_md --output-format tei --output out.xml

# SVG dependency tree (UD-Kanbun)
flexipipe process --backend udkanbun --data '不入虎穴不得虎子' --output-format svg --svg-style tree --output tree.svg

# Arrow-style dependency viz
flexipipe process --language nld --example udhr --backend spacy --output-format svg --svg-style dep --output deps.svg

# UDPipe REST, raw mode
flexipipe process --backend udpipe --udpipe-model english-ewt-ud-2.15-241121 --input-format raw --input story.txt

# Chain backends: UDPipe then NameTag for NER
echo "Mary bought a new bicycle in Germany." | \
  flexipipe process --backend udpipe | \
  flexipipe process --backend nametag --output-format conllu-ne
```

**Important switches:** `--backend` (use `flexipipe info backends` to see options), `--model` / `--language`, `--download-model`, `--output-format` (tei, conllu, conllu-ne, json, svg), `--svg-style`, `--create-implicit-mwt`. Run `flexipipe process --help` for full list.

---

## Benchmarking

```bash
# Single backend/model test
flexipipe benchmark --test --test-file UD_English-EWT/en_ewt-ud-test.conllu --backend spacy --model en_core_web_trf --mode tokenized --verbose

# Sweep across languages/backends
flexipipe benchmark --run --languages en nl de --backends spacy stanza

# Show stored results
flexipipe benchmark --show --sort-by upos
```

`--mode`: how gold data is fed (`raw`, `tokenized`, `split`, `auto`).

---

## Training

```bash
# Train flexitag on UD treebank
flexipipe train --backend flexitag --ud-data /path/to/UD_English-EWT --output-dir models/flexitag-en

# Backend-owned training (where implemented)
flexipipe train --backend spacy --model en_core_web_md --train-data data/train.conllu --dev-data data/dev.conllu --output-dir models/spacy-en
```

---

## Configuration

```bash
# Set default backend, output format, models dir, implicit MWT
flexipipe config --set-models-dir /path/to/models --set-default-backend spacy --set-default-output-format conllu --set-default-create-implicit-mwt true

# Inspect config
flexipipe config --show
```
