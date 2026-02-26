# Backends Overview

flexiPipe is a **meta-tool**: it does not implement one fixed NLP pipeline but instead provides a uniform interface to many existing pipelines and toolkits (flexitag, SpaCy, Stanza, Flair, HuggingFace Transformers, UDPipe, TreeTagger, NameTag, UD-Kanbun, and others). You run the same commands and options regardless of which backend is doing the work; input and output formats stay consistent (CoNLL-U, TEITOK, raw text, etc.).

**Most backends are not installed by default.** Only the core runtime and a minimal set of components are included. To add a backend, use **`flexipipe install <name>`** (e.g. `flexipipe install spacy`, `flexipipe install stanza`). Some backends are shipped as separate packages or repos (e.g. Stanza has its own repository and can include third-party backend code); flexiPipe discovers and uses them once they are installed in the environment.

To see what is available on your system and how to use it:

* **`flexipipe info backends`** — list installed backends and their capabilities.
* **`flexipipe info models --backend <name>`** — list models (and often download options) for a given backend.
* **`flexipipe info tasks`**, **`flexipipe info languages`**, **`flexipipe info examples`** — explore tasks, languages, and sample usage.

The table below summarizes the main backends; use **`flexipipe info backends`** for the full, up-to-date list.

| Backend | Mode(s) | Highlights | Notes |
| --- | --- | --- | --- |
| `flexitag` | Raw / tokenized | Built-in Viterbi tagger, rule-based segmentation | Requires flexitag model (`model_vocab.json`). |
| `spacy` | Raw + tokenized | NER, dependency parsing, model discovery & download | Pre-tokenized preserves tokids/MWTs. |
| `stanza` | Raw + tokenized | Full UD pipeline, package selection | `--download-model` or `--model`/`--language`. |
| `flair` | Raw-focused | POS + NER, confidence scores | Works best in raw mode. |
| `transformers` | Raw + tokenized | HuggingFace token-classification (POS/NER), metadata | Requires `--model <huggingface_id>`. |
| `udpipe` | Raw + tokenized | REST, batching | `--udpipe-model`, optional `--udpipe-param`. |
| `udmorph` | Tokenized | REST morph-only | `--udmorph-model`. |
| `treetagger` | Tokenized | Local TreeTagger, lemma + XPOS | Install TreeTagger separately; `--treetagger-model`. |
| `nametag` | Raw + tokenized | REST NER, 21 languages | `--nametag-model` or `--language`. |
| `udkanbun` | Raw | Classical Chinese, dependency parsing, glosses | Install via `flexipipe install udkanbun`. |

---

## Backend sources and references

The **adapter code** that connects each backend to flexiPipe lives in the main repository:

* **flexiPipe backends (built-in adapters):** [github.com/ufal/flexipipe/tree/main/flexipipe/backends](https://github.com/ufal/flexipipe/tree/main/flexipipe/backends)

Third-party backends can register via the `flexipipe.backends` entry point and may be developed in separate repos (e.g. [Stanza](https://github.com/stanfordnlp/stanza) or community packages).

| Backend | Project / source | Notes |
| --- | --- | --- |
| **flexitag** | [flexiPipe repo](https://github.com/ufal/flexipipe) | Built-in rule-based tagger; part of flexiPipe. |
| **spacy** | [spacy.io](https://spacy.io) — *Explosion* | Industrial-strength NLP; see project for authors and citation. |
| **stanza** | [StanfordNLP/Stanza](https://github.com/stanfordnlp/stanza) — *Qi et al.* | [Stanza: A Python NLP Library for Many Human Languages](https://aclanthology.org/2020.acl-demos.14/). |
| **transformers** | [Hugging Face Transformers](https://huggingface.co/docs/transformers) — *Hugging Face* | [Transformers: State-of-the-Art Natural Language Processing](https://aclanthology.org/2020.emnlp-demos.6/). |
| **udpipe** | [Lindat UDPipe (REST)](https://lindat.mff.cuni.cz/services/udpipe) — *ÚFAL* | [Straka & Straková (2017)](https://aclanthology.org/P17-4011/). |
| **udpipe1** | [ufal/udpipe](https://github.com/ufal/udpipe) | Local UDPipe 1.x CLI; same team as above. |
| **udmorph** | [Lindat UDMorph](https://lindat.mff.cuni.cz/services/udmorph) — *ÚFAL* | REST morphological tagging. |
| **nametag** | [Lindat NameTag](https://lindat.mff.cuni.cz/services/nametag) — *ÚFAL* | NER REST service; multiple languages. |
| **treetagger** | [TreeTagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) — *Helmut Schmid, LMU* | Lemma + XPOS; install CLI separately. |
| **udkanbun** | [UD-Kanbun](https://koichiyasuoka.github.io/UD-Kanbun/) — *Yasuoka* | Classical Chinese. |
| **classla** | [CLARIN.SI Classla](https://github.com/clarinsi/classla) | South Slavic NLP. |
| **fasttext** | [fastText](https://fasttext.cc) — *Facebook Research* | Word embeddings / language ID. |

Use **`flexipipe info backends`** to see the live list and each backend’s `url` when available.

---

## Selected backend details

### flexitag (built-in)

Default backend shipped with flexiPipe: rule-based segmentation and a Viterbi-style tagger. Requires a flexitag model (`model_vocab.json`). Supports training. No extra install. See the [flexiPipe repo](https://github.com/ufal/flexipipe) for the implementation.

### SpaCy

Fast NLP library with pre-trained models for many languages. Install with `flexipipe install spacy`. Use **`flexipipe info models --backend spacy`** to list and download models. Supports raw and tokenized input; pre-tokenized input preserves tokids and multi-word tokens. [Project & docs](https://spacy.io).

### Stanza

Stanford NLP library with full UD pipelines. Often installed from its own repo or as a separate package; flexiPipe discovers it via the `flexipipe.backends` entry point. Use `--download-model`, `--model`, or `--language`. [Stanza (Qi et al., ACL 2020)](https://github.com/stanfordnlp/stanza).

### HuggingFace Transformers

Use token-classification models for POS or NER. **`flexipipe info models --backend transformers`** shows models with metadata (tasks, base model, training data). Requires `--model <huggingface_id>`.

```bash
echo "Why do we need an Old French model?" | \
  flexipipe process --backend transformers \
    --model Davlan/bert-base-multilingual-cased-ner-hrl \
    --transformers-device cpu --output-format conllu
```

Key options: `--model` (required), `--transformers-task` (tag/ner), `--transformers-device`, `--transformers-adapter`, `--transformers-revision`, `--transformers-trust-remote-code`. [Transformers docs](https://huggingface.co/docs/transformers).

---

↑ [Documentation index](README.md) · [Home](Home.md) · [Installation](Installation.md) · [Quick Start](Quick-Start.md) · [Reference](Reference.md) · [Contributing](Contributing.md)
