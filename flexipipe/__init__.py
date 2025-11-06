"""
FlexiPipe: Flexible transformer-based NLP pipeline for tagging, parsing, and normalization.

Supports multiple annotation schemes (UD, XPOS-only) and works with both modern and historic texts.
Language-agnostic and fine-tunable for specific corpora.
"""

__version__ = "1.0.0"

from flexipipe.config import FlexiPipeConfig
from flexipipe.tagger import FlexiPipeTagger

__all__ = ['FlexiPipeConfig', 'FlexiPipeTagger', '__version__']

