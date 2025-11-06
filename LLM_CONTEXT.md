# FlexiPipe Development Context for LLMs

## Objective

FlexiPipe is a flexible, transformer-based NLP pipeline for tagging, parsing, and normalization. It supports multiple annotation schemes (UD, XPOS-only, etc.) and is designed to work with both modern and historic texts. The system is language-agnostic and can be fine-tuned for specific corpora.

## Current Status

- **Core functionality**: Complete and working
- **Multi-platform support**: CUDA, MPS (Apple Silicon), and CPU
- **Modular architecture**: Recently refactored from monolithic script
- **Pipeline support**: Can read from stdin and write to stdout
- **Lightweight mode**: Viterbi-only tagging without torch dependency

## Project Structure

```
flexipipe/
├── flexipipe/              # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration classes
│   ├── data_loading.py     # CoNLL-U, TEITOK XML, plain text loaders
│   ├── normalization.py    # Normalization functions
│   ├── contractions.py     # Contraction splitting
│   ├── viterbi.py          # Lightweight Viterbi tagging (NO torch)
│   ├── vocabulary.py       # Vocabulary building and management
│   ├── models.py           # PyTorch models (requires torch)
│   ├── trainer.py          # Training logic (requires torch)
│   ├── tagger.py           # Main FlexiPipeTagger class
│   └── cli/                # Command-line interfaces
│       ├── __init__.py
│       ├── train.py        # Training command
│       ├── tag.py           # Tagging command (pipeline-friendly)
│       ├── analyze.py       # Analysis command
│       ├── viterbi_tag.py  # Lightweight Viterbi-only tagging
│       └── create_vocab.py  # Vocabulary creation
├── tests/                  # Unit tests
├── examples/               # Example scripts
├── dev/                    # ⚠️ TEMPORARY DEVELOPMENT FILES
│                          # Files here are for active development/testing
│                          # Do NOT commit to git - these are temporary
│                          # Use for scripts, experiments, debugging
├── models/                 # ⚠️ TRAINED MODELS AND VOCABULARIES
│                          # Large files - do NOT commit to git
│                          # Contains trained models, vocab JSON files
├── README.md              # User documentation
├── LLM_CONTEXT.md         # This file - context for LLM assistants
└── setup.py               # Package installation

```

## Important Development Guidelines

### Temporary Files Location
- **Use `dev/` for temporary scripts, experiments, and debugging files**
- Files in `dev/` are git-ignored and should be considered disposable
- When working on features, create temporary test scripts in `dev/`
- Clean up `dev/` periodically

### Model Storage
- **Trained models go in `models/`**
- This directory is git-ignored (models are large)
- Vocabulary JSON files also go here
- Never commit model files to git

### Module Dependencies
- **`viterbi.py`**: MUST NOT import torch/transformers (lightweight mode)
- **`models.py`, `trainer.py`**: Require torch/transformers
- **`tagger.py`**: Can work with or without torch (falls back to Viterbi)
- Keep dependencies clear to enable lightweight mode

### Pipeline Support
- Commands should support stdin/stdout for pipeline usage
- Example: `echo "text" | flexipipe tag --vocab vocab.json`
- Use `-` as filename to indicate stdin/stdout
- Check `sys.stdin.isatty()` to detect if input is from terminal or pipe

## Future Plans

1. **Performance optimization**: Further memory optimizations for MPS
2. **Additional formats**: Support for more input/output formats
3. **Documentation**: Expand user documentation with more examples
4. **Testing**: Add comprehensive unit tests
5. **CI/CD**: Set up automated testing and deployment

## Key Design Decisions

- **Modular architecture**: Split from monolithic script for maintainability
- **Lightweight mode**: Viterbi tagging without torch for fast, dependency-free tagging
- **Pipeline-friendly**: Support stdin/stdout for Unix-style pipelines
- **Multi-platform**: Automatic device detection and optimization
- **Language-agnostic**: Works with any language/corpus with proper vocabulary

## Common Tasks

### Adding a new feature
1. Create temporary test script in `dev/`
2. Implement feature in appropriate module
3. Add CLI support if needed
4. Test with examples
5. Clean up `dev/` files

### Debugging
- Use `dev/` for debug scripts
- Enable `--debug` flag for verbose output
- Check logs in `models/*/logs/` for training issues

### Testing
- Unit tests go in `tests/`
- Integration tests can use `dev/` temporarily
- Examples in `examples/` should work out of the box

