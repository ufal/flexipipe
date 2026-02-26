# Contributing

## Project layout

```
flexipipe/              # Main Python package (CLI, backends, converters)
flexitag/               # C++ flexitag sources and bindings
src/                    # Additional C++ helpers
wiki/                   # Wiki page sources (sync to GitHub Wiki via scripts/sync-wiki.sh)
README_CPP.md           # Native build instructions
```

Third-party C++ dependencies (pugixml, rapidjson) are fetched via CMake FetchContent.

## Guidelines

* Respect existing document structures (`tokid`, `Sentence.entities`, `space_after`).
* When adding backends, CLI options, or config keys, update the Wiki (and optionally the concise README) so docs stay in sync.
* The transformers backend is fully implemented with model discovery and metadata. Training hooks exist for flexitag and some neural backends.

Please open issues or discussions for missing features, incomplete documentation, or new backend ideas.

---

↑ [Documentation index](README.md) · [Home](Home.md) · [Installation](Installation.md) · [Quick Start](Quick-Start.md) · [Backends](Backends.md) · [Reference](Reference.md)
