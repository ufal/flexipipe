"""UD-Kanbun backend implementation and registry spec."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backend_spec import BackendSpec
from ..conllu import parse_conllu_from_backend
from ..doc import Document
from ..neural_backend import BackendManager, NeuralResult


def get_udkanbun_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = 60 * 60 * 24,  # 24 hours
    verbose: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Return UD-Kanbun model entries.
    
    UD-Kanbun uses a single default model, so we return a single entry.
    """
    from ..language_utils import build_model_entry
    
    result: Dict[str, Dict[str, str]] = {}
    
    # UD-Kanbun has a single default model for Classical Chinese
    entry = build_model_entry(
        "udkanbun",
        "default",
        language_code="lzh",
        language_name="Classical Chinese",
        preferred=True,
        components=["tokenizer", "tagger", "parser"],
        description="UD-Kanbun default model for Classical Chinese (漢文/文言文)",
    )
    entry["status"] = "available"
    result["default"] = entry
    
    return result


def list_udkanbun_models(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> int:
    """Print available UD-Kanbun models."""
    try:
        entries = get_udkanbun_model_entries(
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            verbose=True,
        )
        print(f"\nAvailable UD-Kanbun models:")
        print(f"{'Model ID':<20} {'ISO':<6} {'Language':<20} {'Status':<12}")
        print("=" * 65)
        for key in sorted(entries.keys()):
            entry = entries[key]
            iso = entry.get("language_iso", "")[:6]
            lang = entry.get("language_name", "")
            status = entry.get("status", "")
            suffix = "*" if entry.get("preferred") else ""
            print(f"{key+suffix:<20} {iso:<6} {lang:<20} {status:<12}")
        preferred_note = "\n(*) Preferred model used by auto-selection"
        print(preferred_note)
        print(f"\nTotal: {len(entries)} model(s)")
        print("UD-Kanbun models are loaded automatically on first use")
        print("Features: tokenization, lemma, upos, xpos, feats, depparse")
        return 0
    except Exception as exc:
        print(f"Error listing UD-Kanbun models: {exc}")
        import traceback
        traceback.print_exc()
        return 1


class UDKanbunBackend(BackendManager):
    """UD-Kanbun backend for Classical Chinese (漢文/文言文)."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        mecab: bool = True,
        danku: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize UD-Kanbun backend using the spaCy version.
        
        Args:
            model_name: Model name (currently only "default" is supported)
            mecab: If True, use MeCab for tokenizer and POS-tagger (default: True)
            danku: If True, try to segment sentences automatically (default: False)
            verbose: Enable verbose output
            debug: Enable debug output (only for --debug flag)
        """
        from ..dependency_utils import ensure_package_installed
        
        # Ensure udkanbun package is installed (with auto-install/prompt support)
        ensure_package_installed(
            "udkanbun",
            module_name="udkanbun",
            friendly_name="UD-Kanbun",
            backend_name="udkanbun",
        )
        
        # Also ensure spaCy is available (required for udkanbun.spacy)
        from ..dependency_utils import ensure_extra_installed
        ensure_extra_installed("spacy", module_name="spacy", friendly_name="SpaCy")
        
        import udkanbun.spacy
        
        self._udkanbun_spacy = udkanbun.spacy
        self._model_name = model_name or "default"
        self._mecab = mecab
        self._danku = danku
        self._verbose = verbose
        self._debug = debug
        self._pipeline = None

    def _ensure_pipeline(self):
        """Ensure the UD-Kanbun spaCy pipeline is loaded."""
        if self._pipeline:
            return self._pipeline
        
        # UD-Kanbun spaCy API uses MeCab and Danku (capitalized)
        self._pipeline = self._udkanbun_spacy.load(MeCab=self._mecab, Danku=self._danku)
        return self._pipeline

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        """
        Tag a document using UD-Kanbun.
        
        Args:
            document: Input document
            overrides: Optional overrides (not used)
            preserve_pos_tags: If True, preserve existing POS tags (not supported)
            components: Optional list of components (not used, all components are always run)
            use_raw_text: If True, use raw text from sentences
        """
        del overrides, preserve_pos_tags, components
        
        pipeline = self._ensure_pipeline()
        start_time = time.time()

        # Convert document to text
        if use_raw_text or not document.sentences:
            # Use raw text from sentences
            text = "\n".join(sentence.text for sentence in document.sentences if sentence.text)
        else:
            # Reconstruct text from tokens
            sentences = []
            for sentence in document.sentences:
                if sentence.text:
                    sentences.append(sentence.text)
                else:
                    # Reconstruct from tokens
                    tokens = []
                    for token in sentence.tokens:
                        if token.is_mwt and token.subtokens:
                            tokens.extend([st.form for st in token.subtokens])
                        else:
                            tokens.append(token.form)
                    sentences.append("".join(tokens))  # Classical Chinese typically has no spaces
            text = "\n".join(sentences)

        if not text.strip():
            return NeuralResult(
                document=document,
                stats={
                    "backend": "udkanbun",
                    "token_count": 0,
                    "model": self._model_name,
                },
            )

        # Process text with UD-Kanbun spaCy version
        # Use spaCy Doc directly and convert using existing spaCy backend conversion
        from ..spacy_backend import _spacy_doc_to_document
        
        # Process text line by line (UD-Kanbun processes line by line)
        all_spacy_docs = []
        for line in text.split("\n"):
            if not line.strip():
                continue
            
            # Process the line with UD-Kanbun spaCy pipeline
            spacy_doc = pipeline(line.strip())
            all_spacy_docs.append(spacy_doc)
        
        # Convert spaCy Docs to flexipipe Document
        # Start with first sentence
        if not all_spacy_docs:
            # Empty input - return empty document
            tagged_doc = Document(id=document.id, meta=dict(document.meta))
            return NeuralResult(document=tagged_doc, stats={})
        
        tagged_doc = _spacy_doc_to_document(all_spacy_docs[0], document, force_single_sentence=True)
        
        # Store spaCy Docs in document meta for SVG output support
        # This allows calling to_svg() on the original spaCy Doc objects
        # Note: Currently falls back to displacy if UD-Kanbun doesn't provide native SVG
        tagged_doc.meta["_udkanbun_spacy_docs"] = all_spacy_docs
        
        # If UD-Kanbun provides a custom SVG renderer, it can be registered here:
        # tagged_doc.meta["_svg_renderer"] = lambda: udkanbun_native_svg_renderer(all_spacy_docs)
        
        # Debug: Check converted document structure (only with --debug)
        if self._debug and tagged_doc.sentences:
            sent = tagged_doc.sentences[0]
            print(f"[udkanbun DEBUG] After conversion: {len(sent.tokens)} tokens", file=sys.stderr)
            for i, tok in enumerate(sent.tokens[:3]):
                print(f"[udkanbun DEBUG] Token {i+1}: id={tok.id}, form={tok.form}, is_mwt={tok.is_mwt}, head={getattr(tok, 'head', None)}, deprel={getattr(tok, 'deprel', None)}", file=sys.stderr)
        
        # Add remaining sentences if any
        for spacy_doc in all_spacy_docs[1:]:
            additional_doc = _spacy_doc_to_document(spacy_doc, None, force_single_sentence=True)
            tagged_doc.sentences.extend(additional_doc.sentences)
        
        # Extract MISC fields (including Gloss) from UD-Kanbun's CoNLL-U output
        # UD-Kanbun stores Gloss and other MISC fields in the CoNLL-U format
        # We need to extract these and merge them into our tokens
        for sent_idx, spacy_doc in enumerate(all_spacy_docs):
            if sent_idx >= len(tagged_doc.sentences):
                break
            
            # Get CoNLL-U output for this sentence
            conllu_str = self._udkanbun_spacy.to_conllu(spacy_doc).strip()
            
            # Parse MISC fields from CoNLL-U lines
            # Format: ID\tFORM\tLEMMA\tUPOS\tXPOS\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC
            misc_by_id = {}
            for line in conllu_str.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 10:
                    try:
                        token_id = int(parts[0])
                        misc_field = parts[9] if parts[9] != "_" else ""
                        if misc_field:
                            misc_by_id[token_id] = misc_field
                    except (ValueError, IndexError):
                        continue
            
            # Apply MISC fields to tokens in the sentence
            sent = tagged_doc.sentences[sent_idx]
            for token in sent.tokens:
                if token.id in misc_by_id:
                    token.misc = misc_by_id[token.id]

        # UD-Kanbun outputs proper tokenization and dependency relations
        # Never apply create_implicit_mwt for UD-Kanbun - it would break the dependency structure
        tagged_doc.meta["_disable_create_implicit_mwt"] = True
        
        elapsed = time.time() - start_time
        token_count = sum(len(sent.tokens) for sent in tagged_doc.sentences)
        stats = {
            "backend": "udkanbun",
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
            "sentences_per_second": len(tagged_doc.sentences) / elapsed if elapsed > 0 else 0.0,
            "model": self._model_name,
        }
        return NeuralResult(document=tagged_doc, stats=stats)

    def train(
        self,
        train_data: Any,
        output_dir: Path,
        *,
        dev_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> Path:
        """Training is not supported by UD-Kanbun."""
        raise NotImplementedError("UD-Kanbun backend does not support training.")

    def supports_training(self) -> bool:
        return False


def _create_udkanbun_backend(
    *,
    model_name: Optional[str] = None,
    mecab: bool = True,
    danku: bool = False,
    verbose: bool = False,
    debug: bool = False,
    download_model: bool = False,
    training: bool = False,
    **kwargs: Any,
) -> UDKanbunBackend:
    """Factory function to create a UD-Kanbun backend."""
    from ..backend_utils import validate_backend_kwargs
    
    # UD-Kanbun loads models automatically, so download_model is ignored
    # training is not supported but allowed to avoid errors
    # language is passed by CLI but not used by UD-Kanbun (it's always Classical Chinese)
    validate_backend_kwargs(kwargs, "UD-Kanbun", allowed_extra=["download_model", "training", "language"])
    
    return UDKanbunBackend(
        model_name=model_name,
        mecab=mecab,
        danku=danku,
        verbose=verbose,
        debug=debug,
    )


BACKEND_SPEC = BackendSpec(
    name="udkanbun",
    description="UD-Kanbun - Tokenizer, POS-Tagger, and Dependency-Parser for Classical Chinese (漢文/文言文)",
    factory=_create_udkanbun_backend,
    get_model_entries=get_udkanbun_model_entries,
    list_models=list_udkanbun_models,
    supports_training=False,
    is_rest=False,
    url="https://koichiyasuoka.github.io/UD-Kanbun/",
    install_instructions="Install via: pip install udkanbun",
)
