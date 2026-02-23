# Backends Overview

Use **`flexipipe info backends`** to see available backends. Install more with `flexipipe install <name>` (e.g. `flexipipe install spacy`).

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

## HuggingFace Transformers backend

Use token-classification models for POS or NER. `flexipipe info models --backend transformers` shows models with metadata (tasks, base model, training data).

```bash
echo "Why do we need an Old French model?" | \
  flexipipe process --backend transformers \
    --model Davlan/bert-base-multilingual-cased-ner-hrl \
    --transformers-device cpu --output-format conllu
```

Key options: `--model` (required), `--transformers-task` (tag/ner), `--transformers-device`, `--transformers-adapter`, `--transformers-revision`, `--transformers-trust-remote-code`.
