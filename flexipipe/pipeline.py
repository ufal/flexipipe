from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Union

from .doc import Document, Sentence, Token
from .engine import FlexitagFallback, FlexitagResult
from .neural_backend import BackendManager, NeuralResult
try:
    from . import bert
except ImportError:
    bert = None  # bert module is optional/deprecated


@dataclass
class PipelineConfig:
    params_file: Optional[str] = None  # Optional if using neural backend only
    neural_backend: Optional[BackendManager] = None
    bert_model: Optional[str] = None
    flexitag_options: Dict[str, object] = field(default_factory=dict)
    split_with_flexitag: bool = True
    tag_fallback_threshold: float = 0.75
    use_neural_as_primary: bool = False  # If True, use neural backend as primary, flexitag as fallback
    preserve_pos_tags: bool = False  # If True, preserve existing POS tags (for two-stage tagging)
    neural_components: Optional[List[str]] = None  # Components to run (e.g., ["parser"] for parsing only)
    debug: bool = False


class FlexiPipeline:
    """Hybrid pipeline combining neural backends with the flexitag fallback."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._debug = config.debug
        
        # Initialize flexitag fallback if params_file provided
        if config.params_file:
            self.fallback = FlexitagFallback(
                config.params_file,
                options=config.flexitag_options,
                debug=config.debug,
            )
        else:
            self.fallback = None
        
        # Initialize neural backend if provided
        self.neural_backend = config.neural_backend
        
        # Legacy BERT component (deprecated, use neural_backend instead)
        self.bert_component = bert.BertComponent.from_model(config.bert_model) if config.bert_model else None

    def process(self, document: Document) -> Union[FlexitagResult, NeuralResult]:
        doc = Document.from_dict(document.to_dict())  # defensive copy

        if self._debug:
            sent_count = len(doc.sentences)
            tok_count = sum(len(sent.tokens) for sent in doc.sentences)
            print(
                f"[flexipipe] processing document id={doc.id or '<unnamed>'} "
                f"sentences={sent_count} tokens={tok_count}"
            )

        # Use neural backend as primary if configured
        if self.config.use_neural_as_primary and self.neural_backend:
            # Check if we should preserve POS tags (two-stage tagging workflow)
            preserve_pos = getattr(self.config, 'preserve_pos_tags', False)
            components = getattr(self.config, 'neural_components', None)
            neural_result = self.neural_backend.tag(
                doc,
                preserve_pos_tags=preserve_pos,
                components=components
            )
            
            # Optionally use flexitag as fallback for uncertain predictions
            if self.fallback and neural_result.confidence_scores:
                # Filter tokens with low confidence
                uncertain_tokens = [
                    (sent_idx, tok_idx)
                    for sent_idx, sent in enumerate(neural_result.document.sentences)
                    for tok_idx, score in enumerate(neural_result.confidence_scores)
                    if score < self.config.tag_fallback_threshold
                ]
                
                if uncertain_tokens:
                    if self._debug:
                        print(f"[flexipipe] falling back to flexitag for {len(uncertain_tokens)} uncertain tokens")
                    # Use flexitag for uncertain tokens
                    flexi_result = self.fallback.tag(neural_result.document)
                    # Merge results (neural for confident, flexitag for uncertain)
                    merged = self._merge_neural_flexitag(neural_result.document, flexi_result.document, uncertain_tokens)
                    return FlexitagResult(document=merged, stats=neural_result.stats)
            
            # Convert NeuralResult to FlexitagResult for compatibility
            return FlexitagResult(document=neural_result.document, stats=neural_result.stats)

        # Legacy: Use BERT component if available
        if self.bert_component:
            if self.config.split_with_flexitag:
                doc = self._ensure_split(doc)

            confident, uncertain = self.bert_component.predict(doc)

            if not uncertain:
                return FlexitagResult(document=confident, stats={"fallback_used": 0})

            merged = self._merge_predictions(confident, uncertain)
            if not self.fallback:
                raise RuntimeError("flexitag fallback required but params_file not provided")
            flexi_result = self.fallback.tag(merged)
            flexi_result.stats.setdefault("fallback_used", 0)
            flexi_result.stats["fallback_used"] += 1
            return flexi_result

        # Default: Use flexitag only
        if not self.fallback:
            raise RuntimeError("No backend configured. Provide either params_file or neural_backend.")
        
        if self.config.split_with_flexitag:
            doc = self._ensure_split(doc)
        
        return self.fallback.tag(doc)

    def _ensure_split(self, document: Document) -> Document:
        needs_split = any(tok.is_mwt for tok in document.tokens())
        if not needs_split:
            return document
        if self._debug:
            print("[flexipipe] falling back to flexitag for contraction splitting")
        result = self.fallback.tag(document)
        return result.document

    @staticmethod
    def _merge_predictions(confident: Document, fallback_input: Document) -> Document:
        # For now we simply return the fallback input; sophisticated merging logic can be added later.
        return fallback_input
    
    @staticmethod
    def _merge_neural_flexitag(
        neural_doc: Document,
        flexitag_doc: Document,
        uncertain_tokens: list[tuple[int, int]]
    ) -> Document:
        """Merge neural and flexitag results, using flexitag for uncertain tokens."""
        merged = Document(id=neural_doc.id, meta=dict(neural_doc.meta))
        
        for sent_idx, (neural_sent, flexitag_sent) in enumerate(zip(neural_doc.sentences, flexitag_doc.sentences)):
            merged_sent = Sentence(
                id=neural_sent.id,
                sent_id=neural_sent.sent_id,
                text=neural_sent.text,
                tokens=[],
            )
            
            for tok_idx, (neural_tok, flexitag_tok) in enumerate(zip(neural_sent.tokens, flexitag_sent.tokens)):
                if (sent_idx, tok_idx) in uncertain_tokens:
                    # Use flexitag result for uncertain tokens
                    merged_sent.tokens.append(flexitag_tok)
                else:
                    # Use neural result for confident tokens
                    merged_sent.tokens.append(neural_tok)
            
            merged.sentences.append(merged_sent)
        
        return merged
