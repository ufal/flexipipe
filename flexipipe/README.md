# Flexipipe User Guide

Flexipipe is a unified command-line tool for natural language processing that integrates multiple NLP backends (SpaCy, Stanza, ClassLA, Flair, UDPipe, UDMorph, NameTag, and the built-in Flexitag) into a single, easy-to-use interface. It supports multiple input and output formats, automatic language detection, and comprehensive benchmarking capabilities.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Main Commands](#main-commands)
  - [Processing Text](#processing-text)
  - [Information & Discovery](#information--discovery)
  - [Configuration](#configuration)
  - [Training Models](#training-models)
  - [Format Conversion](#format-conversion)
  - [Benchmarking](#benchmarking)
- [Backends](#backends)
- [Input & Output Formats](#input--output-formats)
- [Advanced Features](#advanced-features)
- [Examples](#examples)

---

## Installation

### Requirements

- **Python 3.11+**
- Required Python packages (install via `pip install -r requirements.txt`)

### Optional Backend Dependencies

Install backend-specific packages as needed:

```bash
# For SpaCy backend
pip install spacy

# For Stanza backend
pip install stanza

# For ClassLA backend (South Slavic languages)
pip install classla

# For Flair backend
pip install flair

# REST backends (UDPipe, UDMorph, NameTag) work out of the box
# No additional installation needed
```

### Building Native Components

If you need the high-performance C++ components (flexitag, tokenizer), see the main `README_CPP.md` for build instructions.

---

## Quick Start

### Basic Text Processing

```bash
# Process text from stdin with default backend
echo "Hello, world! This is a test." | python -m flexipipe process

# Process a file with a specific backend
python -m flexipipe process \
  --input document.txt \
  --backend spacy \
  --model en_core_web_sm \
  --output result.conllu

# Process inline text
python -m flexipipe process \
  --data "This is a test sentence." \
  --backend stanza \
  --language en
```

### Discovering Available Models

```bash
# List all available backends
python -m flexipipe info backends

# List models for a specific backend
python -m flexipipe info models --backend spacy

# Search for models by language
python -m flexipipe info models --language nl

# Get JSON output for scripting
python -m flexipipe info models --backend spacy --output-format json
```

---

## Main Commands

### Processing Text

The `process` command is the main entry point for NLP tasks. It handles tokenization, tagging, parsing, lemmatization, normalization, and named entity recognition.

#### Basic Usage

```bash
python -m flexipipe process [OPTIONS]
```

#### Key Options

| Option | Description |
|--------|-------------|
| `--input FILE` | Input file (TEITOK XML, CoNLL-U, or raw text). Omit to read from STDIN. |
| `--output FILE` | Output file. Omit to write to stdout. |
| `--backend NAME` | Backend to use: `flexitag`, `spacy`, `stanza`, `classla`, `flair`, `udpipe`, `udmorph`, `nametag` |
| `--model NAME` | Model name or path (e.g., `en_core_web_sm` for SpaCy, `cs_cac` for Stanza) |
| `--language CODE` | Language code (e.g., `en`, `nl`, `es`). Used for auto-selection or blank models. |
| `--data TEXT` | Provide text directly on command line instead of `--input` |
| `--output-format FORMAT` | Output format: `tei`, `conllu`, or `conllu-ne` |
| `--input-format FORMAT` | Input format: `auto`, `tei`, `conllu`, `conllu-ne`, or `raw` |
| `--tasks TASKS` | Comma-separated tasks: `tokenize`, `segment`, `lemmatize`, `tag`, `parse`, `normalize`, `ner` |
| `--download-model` | Automatically download missing models |
| `--endpoint-url URL` | REST backend endpoint URL (for UDPipe, UDMorph, NameTag) |
| `--timeout SECONDS` | Timeout for REST requests (default: 30) |
| `--params KEY=VALUE` | Additional parameters for REST backends (can be repeated) |

#### Examples

```bash
# Process raw text with SpaCy
python -m flexipipe process \
  --input text.txt \
  --backend spacy \
  --model en_core_web_sm \
  --output-format conllu

# Process TEITOK XML with Stanza
python -m flexipipe process \
  --input document.xml \
  --backend stanza \
  --model cs_cac \
  --output-format tei \
  --output result.xml

# Process with automatic language detection
echo "Bonjour le monde" | python -m flexipipe process \
  --backend spacy \
  --download-model

# Use UDPipe REST service
python -m flexipipe process \
  --input text.txt \
  --backend udpipe \
  --model english-ewt-ud-2.15-241121 \
  --endpoint-url https://lindat.mff.cuni.cz/services/udpipe/api/process

# Process only specific tasks
python -m flexipipe process \
  --input text.txt \
  --backend spacy \
  --tasks tokenize,tag,ner \
  --output result.conllu
```

#### TEITOK-Specific Options

| Option | Description |
|--------|-------------|
| `--attrs-map ATTR:VALUES` | Map TEITOK attributes (e.g., `xpos:msd,pos` or `reg:nform`). Can be repeated. |
| `--tokenize` | Enable tokenization for non-tokenized TEITOK XML files |
| `--textnode XPATH` | XPath to locate text node (default: `.//text`) |
| `--writeback` | Update original TEITOK XML file in-place |
| `--nlpform {form,reg}` | Use original form or normalization (reg) for NLP processing |

#### Advanced Options

| Option | Description |
|--------|-------------|
| `--pretokenize` | Segment and tokenize locally before sending to backend |
| `--create-implicit-mwt` | Create implicit multi-word tokens from SpaceAfter=No sequences |
| `--normalization-style STYLE` | Normalization style: `conservative`, `aggressive`, `enhanced`, or `balanced` |
| `--map-tags-model FILE` | Model vocab JSON for tag mapping (can be repeated) |
| `--map-direction DIR` | Tag mapping direction: `xpos`, `upos-feats`, or `both` |
| `--extra-vocab FILE` | Additional vocabulary files to merge (can be repeated) |

---

### Information & Discovery

The `info` command helps you discover available backends and models.

#### List Backends

```bash
# List all available backends
python -m flexipipe info backends

# JSON output
python -m flexipipe info backends --output-format json
```

#### List Models

```bash
# List models for a specific backend
python -m flexipipe info models --backend spacy

# Search across all backends by language
python -m flexipipe info models --language nl

# Refresh cached model lists
python -m flexipipe info models --backend spacy --refresh-cache

# JSON output
python -m flexipipe info models --backend spacy --output-format json
```

#### Detect Language

```bash
# Detect language from input text
echo "This is a test" | python -m flexipipe info --detect-language

# With minimum length requirement
python -m flexipipe info --detect-language --input text.txt --min-length 20
```

---

### Configuration

The `config` command manages flexipipe settings stored in `~/.flexipipe/config.json`.

#### View Current Configuration

```bash
python -m flexipipe config --show
```

#### Set Configuration Values

```bash
# Set models directory
python -m flexipipe config --set-models-dir /path/to/models

# Set default backend
python -m flexipipe config --set-default-backend spacy

# Set default output format
python -m flexipipe config --set-default-output-format conllu

# Set default implicit MWT creation
python -m flexipipe config --set-default-create-implicit-mwt true
```

#### Environment Variables

You can also override the models directory using an environment variable:

```bash
export FLEXIPIPE_MODELS_DIR=/path/to/models
```

---

### Training Models

The `train` command trains models from training data.

#### Flexitag Training

```bash
# Train from UD treebank directory
python -m flexipipe train \
  --backend flexitag \
  --ud-data /path/to/UD_English-EWT \
  --output-dir models/flexitag-en

# Train from CoNLL-U files
python -m flexipipe train \
  --backend flexitag \
  --train-data train.conllu \
  --dev-data dev.conllu \
  --output-dir models/flexitag-en
```

#### Neural Backend Training

```bash
# Train SpaCy model
python -m flexipipe train \
  --backend spacy \
  --model en_core_web_md \
  --train-data train.conllu \
  --dev-data dev.conllu \
  --output-dir models/spacy-en

# Train Stanza model
python -m flexipipe train \
  --backend stanza \
  --language en \
  --train-data train.conllu \
  --dev-data dev.conllu \
  --output-dir models/stanza-en
```

---

### Format Conversion

The `convert` command converts between different formats.

#### Convert TEITOK to CoNLL-U

```bash
# Convert single file
python -m flexipipe convert \
  --type treebank \
  --input corpus.xml \
  --output corpus.conllu

# Convert entire treebank directory
python -m flexipipe convert \
  --type treebank \
  --input /path/to/teitok/corpus \
  --output /path/to/output \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1
```

#### Convert Lexicon

```bash
python -m flexipipe convert \
  --type lexicon \
  --input lexicon.xml \
  --output model_vocab.json
```

#### Convert Tagged Files

```bash
python -m flexipipe convert \
  --type tagged \
  --input input.conllu \
  --output output.xml \
  --input-format conllu \
  --output-format tei
```

---

### Benchmarking

The `benchmark` command evaluates backend/model combinations against Universal Dependencies treebanks.

#### List Available Tests

```bash
# List all languages with models and/or treebanks
python -m flexipipe benchmark --list-tests

# List available models
python -m flexipipe info models

# List available treebanks
python -m flexipipe benchmark --list-treebanks
```

#### Run Benchmarks

```bash
# Test a single backend/model combination
python -m flexipipe benchmark --test \
  --backend spacy \
  --model en_core_web_sm \
  --treebank UD_English-EWT/en_ewt-ud-test.conllu

# Run benchmarks for a specific language
python -m flexipipe benchmark --run \
  --languages nl \
  --backends spacy stanza

# Run all available combinations
python -m flexipipe benchmark --run \
  --languages all \
  --backends all \
  --download-models

# Run with specific mode
python -m flexipipe benchmark --run \
  --languages nl \
  --backends all \
  --mode tokenized
```

#### View Results

```bash
# Show all results
python -m flexipipe benchmark --show

# Show results for a specific language
python -m flexipipe benchmark --show --language nl

# Show averaged results
python -m flexipipe benchmark --show --average

# Sort by specific metric
python -m flexipipe benchmark --show --average --sort lemma

# JSON output
python -m flexipipe benchmark --show --output-format json
```

#### Benchmark Options

| Option | Description |
|--------|-------------|
| `--treebank-root PATH` | Root directory containing UD treebanks |
| `--run` | Run benchmark calculations |
| `--test` | Test single backend/model/treebank combination |
| `--show` | Display benchmark results |
| `--languages LANG ...` | Languages to benchmark (or `all`) |
| `--backends BACKEND ...` | Backends to benchmark (or `all`) |
| `--mode MODE` | Execution mode: `raw`, `tokenized`, or `split` |
| `--download-models` | Automatically download missing models |
| `--force` | Overwrite existing results |
| `--sort METRIC` | Sort results by metric (e.g., `upos`, `lemma`, `uas`) |
| `--average` | Show averaged results across treebanks |

---

## Backends

Flexipipe supports multiple NLP backends, each with different strengths:

### Flexitag

Built-in rule-based tagger with Viterbi decoding. Fast and lightweight, ideal for languages with limited neural model support.

```bash
python -m flexipipe process \
  --backend flexitag \
  --model /path/to/flexitag/model \
  --input text.txt
```

**Features:**
- Rule-based sentence segmentation
- UD-style tokenization
- Lexicon-aware tagging
- Normalization support

### SpaCy

Fast, production-ready NLP library with pre-trained models for many languages.

```bash
python -m flexipipe process \
  --backend spacy \
  --model en_core_web_sm \
  --input text.txt
```

**Features:**
- Named entity recognition
- Dependency parsing
- Automatic model discovery
- Download-on-demand models
- Supports both raw and tokenized input

**Model Naming:**
- Use model names like `en_core_web_sm`, `nl_core_news_md`
- Or specify language: `--language en` (uses default model if installed)

### Stanza

Stanford NLP library with high-quality models for many languages.

```bash
python -m flexipipe process \
  --backend stanza \
  --model cs_cac \
  --input text.txt
```

**Features:**
- Full UD pipeline (tokenization, tagging, parsing)
- Package selection (e.g., `cs_cac` for Czech)
- Supports raw and tokenized input
- Download-on-demand models

**Model Naming:**
- Use `lang_package` format: `--model cs_cac` (Czech with CAC package)
- Or use language: `--language cs` (uses default package)

### ClassLA

Fork of Stanza optimized for South Slavic languages (Bulgarian, Croatian, Macedonian, Serbian, Slovenian).

```bash
python -m flexipipe process \
  --backend classla \
  --model bg-standard \
  --input text.txt
```

**Features:**
- Optimized for South Slavic languages
- Standard and nonstandard variants
- Automatic model downloads

**Model Naming:**
- Use `lang-type` format: `--model bg-standard`, `sr-nonstandard`

### Flair

State-of-the-art sequence tagging library with multi-task models.

```bash
python -m flexipipe process \
  --backend flair \
  --language en \
  --input text.txt
```

**Features:**
- Multi-task taggers (POS + NER)
- Confidence scores
- Automatic contraction handling
- Works best with raw input

### UDPipe (REST)

REST service for UDPipe models hosted by Lindat MFF CUNI.

```bash
python -m flexipipe process \
  --backend udpipe \
  --model english-ewt-ud-2.15-241121 \
  --input text.txt
```

**Features:**
- Tokenization, tagging, parsing
- Many language models available
- No local installation required
- Default endpoint: Lindat MFF CUNI service

### UDMorph (REST)

REST service for morphological tagging (no dependency parsing).

```bash
python -m flexipipe process \
  --backend udmorph \
  --model czech-pdt-ud-2.15-241121 \
  --input text.txt
```

**Features:**
- Morphological tagging only
- Faster than full parsing
- Many language models available

### NameTag (REST)

REST service for named entity recognition.

```bash
python -m flexipipe process \
  --backend nametag \
  --language en \
  --input text.txt
```

**Features:**
- Named entity recognition
- Supports 21 languages
- Can be combined with other backends

---

## Input & Output Formats

### Input Formats

Flexipipe supports multiple input formats with automatic detection:

#### Raw Text

Plain text files or stdin. The backend handles tokenization.

```bash
echo "Hello world" | python -m flexipipe process --input-format raw
```

#### CoNLL-U

Universal Dependencies CoNLL-U format. Existing annotations are preserved.

```bash
python -m flexipipe process \
  --input file.conllu \
  --input-format conllu
```

#### TEITOK XML

TEITOK XML format with `<tok>` and `<dtok>` elements. Supports custom attribute mappings.

```bash
python -m flexipipe process \
  --input file.xml \
  --input-format tei \
  --attrs-map xpos:msd,pos \
  --attrs-map reg:nform
```

**Attribute Mapping:**
- Use `--attrs-map` to specify which attributes to use
- Format: `attr:value1,value2` (e.g., `xpos:msd,pos`)
- Supported attributes: `xpos`, `reg`, `expan`, `lemma`, `tokid`

### Output Formats

#### CoNLL-U

Standard Universal Dependencies format.

```bash
python -m flexipipe process \
  --input text.txt \
  --output-format conllu \
  --output result.conllu
```

**Features:**
- Optional implicit MWT creation (`--create-implicit-mwt`)
- Named entity encoding (`conllu-ne` format)
- Preserves original token IDs when available

#### TEITOK XML

TEITOK XML format with nested structures.

```bash
python -m flexipipe process \
  --input text.txt \
  --output-format tei \
  --output result.xml
```

**Features:**
- Nested `<tok>`/`<dtok>` structures
- Named entity spans as `<name>` elements
- In-place updates with `--writeback`

---

## Advanced Features

### Automatic Language Detection

Flexipipe can automatically detect the language of input text and select an appropriate backend/model.

```bash
# Automatic detection and selection
echo "Bonjour le monde" | python -m flexipipe process

# Explicit detection
python -m flexipipe info --detect-language --input text.txt
```

### Tag Mapping

Enrich documents with tags from vocabulary files.

```bash
python -m flexipipe process \
  --input file.xml \
  --map-tags-model vocab1.json \
  --map-tags-model vocab2.json \
  --map-direction both \
  --fill-xpos \
  --fill-upos
```

**Options:**
- `--map-tags-model FILE`: Vocabulary JSON file (can be repeated)
- `--map-direction DIR`: `xpos`, `upos-feats`, or `both`
- `--fill-xpos`, `--fill-upos`, `--fill-feats`: Enable specific tag filling
- `--allow-partial`: Allow partial feature matches

### Normalization Styles

Control how normalization is applied:

- **conservative**: Only explicit mappings from vocabulary
- **aggressive**: Pattern-based substitutions
- **enhanced**: Morphological variations
- **balanced**: Combination approach

```bash
python -m flexipipe process \
  --input text.txt \
  --backend flexitag \
  --normalization-style enhanced
```

### Task Selection

Control which NLP tasks are performed:

```bash
# Only tokenization and tagging
python -m flexipipe process \
  --input text.txt \
  --tasks tokenize,tag

# All tasks except parsing
python -m flexipipe process \
  --input text.txt \
  --tasks tokenize,segment,lemmatize,tag,normalize,ner
```

Available tasks: `tokenize`, `segment`, `lemmatize`, `tag`, `parse`, `normalize`, `ner`

---

## Examples

### Example 1: Basic Text Processing

```bash
# Process English text with SpaCy
echo "The quick brown fox jumps over the lazy dog." | \
  python -m flexipipe process \
    --backend spacy \
    --model en_core_web_sm \
    --output-format conllu
```

### Example 2: Processing a TEITOK Corpus

```bash
# Process TEITOK XML with custom attribute mappings
python -m flexipipe process \
  --input corpus.xml \
  --backend stanza \
  --model nl_alpino \
  --attrs-map xpos:msd \
  --attrs-map reg:nform \
  --output-format tei \
  --output result.xml
```

### Example 3: Batch Processing with UDPipe

```bash
# Process multiple files with UDPipe REST
for file in *.txt; do
  python -m flexipipe process \
    --input "$file" \
    --backend udpipe \
    --model english-ewt-ud-2.15-241121 \
    --output "${file%.txt}.conllu"
done
```

### Example 4: Language-Specific Processing

```bash
# Process Czech text with ClassLA
python -m flexipipe process \
  --input czech.txt \
  --backend classla \
  --model cs-standard \
  --output-format conllu

# Process Bulgarian with ClassLA
python -m flexipipe process \
  --input bulgarian.txt \
  --backend classla \
  --model bg-standard \
  --output-format conllu
```

### Example 5: Benchmarking Workflow

```bash
# 1. List available tests
python -m flexipipe benchmark --list-tests

# 2. Run benchmarks for Dutch
python -m flexipipe benchmark --run \
  --languages nl \
  --backends all \
  --download-models

# 3. View results
python -m flexipipe benchmark --show \
  --language nl \
  --average \
  --sort upos
```

### Example 6: Training a Flexitag Model

```bash
# Train from UD treebank
python -m flexipipe train \
  --backend flexitag \
  --ud-data /path/to/UD_English-EWT \
  --output-dir models/flexitag-en

# Use the trained model
python -m flexipipe process \
  --input text.txt \
  --backend flexitag \
  --model models/flexitag-en
```

### Example 7: Format Conversion

```bash
# Convert TEITOK corpus to UD format
python -m flexipipe convert \
  --type treebank \
  --input /path/to/teitok/corpus \
  --output /path/to/ud/corpus \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1
```

### Example 8: Combining Backends

```bash
# Use SpaCy for main processing, NameTag for NER
python -m flexipipe process \
  --input text.txt \
  --backend spacy \
  --model en_core_web_sm \
  --tasks tokenize,tag,parse

# Then add NER with NameTag
python -m flexipipe process \
  --input text.txt \
  --backend nametag \
  --language en \
  --tasks ner
```

---

## Tips & Best Practices

1. **Model Discovery**: Always use `info models` to discover available models before processing
2. **Language Detection**: Let flexipipe auto-detect language when unsure
3. **Model Downloads**: Use `--download-model` for automatic model installation
4. **Format Detection**: Use `--input-format auto` (default) for automatic format detection
5. **Benchmarking**: Use `--list-tests` to see what combinations are available before running benchmarks
6. **Configuration**: Set defaults via `config` command to avoid repeating options
7. **REST Backends**: Use `--endpoint-url` to point to custom REST service instances
8. **TEITOK Processing**: Use `--attrs-map` to handle different TEITOK attribute naming conventions

---

## Getting Help

```bash
# General help
python -m flexipipe --help

# Command-specific help
python -m flexipipe process --help
python -m flexipipe info --help
python -m flexipipe benchmark --help

# List available backends
python -m flexipipe info backends

# List models for a backend
python -m flexipipe info models --backend spacy
```

---

## Troubleshooting

### Model Not Found

If a model is not found, try:
1. Check available models: `python -m flexipipe info models --backend <backend>`
2. Use `--download-model` to automatically download
3. For SpaCy: `python -m spacy download <model>`
4. For Stanza: Models download automatically on first use

### Language Not Detected

If language detection fails:
1. Provide more text (minimum length can be adjusted)
2. Explicitly specify `--language`
3. Check if backend supports the language: `python -m flexipipe info models --language <code>`

### REST Backend Errors

If REST backends fail:
1. Check network connectivity
2. Verify endpoint URL: `--endpoint-url <url>`
3. Increase timeout: `--timeout 60`
4. Use `--debug` for detailed error messages

### Format Detection Issues

If format detection fails:
1. Explicitly specify `--input-format`
2. Check file extension (`.conllu`, `.xml`)
3. Use `--debug` to see detection details

---

For more information, see the main project README or check the source code documentation.

