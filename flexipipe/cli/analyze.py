#!/usr/bin/env python3
"""
Analyze command for FlexiPipe.
"""

import sys
import argparse
import json
from pathlib import Path

from flexipipe.core import (
    FlexiPipeConfig, FlexiPipeTagger, set_conllu_expansion_key,
    load_teitok_xml
)

def main():
    parser = argparse.ArgumentParser(description='FlexiPipe analyze: Analyze vocabulary and suffixes')
    parser.add_argument('--model', type=Path, help='Path to trained model')
    parser.add_argument('--vocab', type=Path, help='Vocabulary JSON to analyze')
    parser.add_argument('--normalization-suffixes', type=Path,
                       help='External suffix list (JSON)')
    parser.add_argument('--expan', default='expan',
                       help='TEITOK attribute name for expansion')
    parser.add_argument('--xpos-attr', default='xpos',
                       help='TEITOK attribute name(s) for XPOS')
    parser.add_argument('--lemma-anchor', choices=['reg', 'form', 'both'], default='both',
                       help='Anchor for deriving inflection suffixes')
    parser.add_argument('--output', type=Path, help='Write analysis JSON to this file (default: stdout)')
    
    args = parser.parse_args()
    
    # Configure
    set_conllu_expansion_key(args.expan)
    load_teitok_xml._xpos_attr = args.xpos_attr
    load_teitok_xml._expan_attr = args.expan
    
    config = FlexiPipeConfig(
        normalize=True,
        conservative_normalization=True,
        normalization_suffixes_file=args.normalization_suffixes,
        lemma_anchor=args.lemma_anchor,
        train_tokenizer=False,
        train_tagger=False,
        train_parser=False,
        train_lemmatizer=False,
        train_normalizer=False,
    )
    
    # Load vocabulary
    vocab = {}
    if args.vocab:
        with open(args.vocab, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
            vocab = vocab_data.get('vocab', {})
        else:
            vocab = vocab_data
    elif args.model and Path(args.model).exists():
        model_vocab_file = Path(args.model) / 'model_vocab.json'
        if model_vocab_file.exists():
            with open(model_vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
                vocab = vocab_data.get('vocab', {})
            else:
                vocab = vocab_data
    
    # Analyze
    tagger = FlexiPipeTagger(config, vocab=vocab, model_path=args.model if args.model else None)
    if not tagger.vocab:
        tagger.vocab = vocab
    tagger._build_normalization_inflection_suffixes()
    suffixes = tagger.inflection_suffixes or []
    
    analysis = {
        'lemma_anchor': config.lemma_anchor,
        'source': 'external' if args.normalization_suffixes else 'derived',
        'num_suffixes': len(suffixes),
        'suffixes': suffixes,
    }
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as out:
            json.dump(analysis, out, ensure_ascii=False, indent=2)
        print(f"Wrote suffix analysis to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(analysis, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

