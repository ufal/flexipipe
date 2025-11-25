"""
Neural backend interface for flexipipe.

This module provides a unified interface for neural NLP models that can be used
as an alternative or complement to the flexitag rule-based tagger.

Backend implementations are in separate modules (e.g., backends/spacy.py, backends/transformers.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .doc import Document


@dataclass
class NeuralResult:
    """Result from neural backend processing."""
    document: Document
    stats: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Optional[List[float]] = None  # Per-token confidence scores


class BackendManager(ABC):
    """Abstract base class for neural backends (SpaCy, Transformers, etc.)."""
    
    @abstractmethod
    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None
    ) -> NeuralResult:
        """
        Tag a document using the backend model.
        
        Args:
            document: Input document
            overrides: Optional overrides for settings
            preserve_pos_tags: If True, preserve existing POS tags from input (for two-stage tagging)
            components: Optional list of components to run (e.g., ["parser"] for parsing only)
        """
        pass
    
    @abstractmethod
    def train(
        self,
        train_data: Union[Document, List[Document], Path],
        output_dir: Path,
        *,
        dev_data: Optional[Union[Document, List[Document], Path]] = None,
        **kwargs
    ) -> Path:
        """Train a model on the provided data."""
        pass
    
    @property
    @abstractmethod
    def supports_training(self) -> bool:
        """Whether this backend supports training."""
        pass




# Factory function for creating backends
# create_backend function has been moved to backend_registry.py
# Import it from there for backwards compatibility
from .backend_registry import create_backend
