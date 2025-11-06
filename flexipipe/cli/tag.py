#!/usr/bin/env python3
"""
Tag command for FlexiPipe - supports stdin/stdout for pipeline usage.
"""

import sys
import argparse
import json
from pathlib import Path

# Import from core module (temporary - will be refactored to use modular imports)
from flexipipe.core import (
    FlexiPipeConfig, FlexiPipeTagger, set_conllu_expansion_key,
    load_teitok_xml, build_vocabulary
)

def main():
    parser = argparse.ArgumentParser(
        description='FlexiPipe tag: Tag text (supports stdin/stdout for pipelines)'
    )
    parser.add_argument('input', nargs='?', type=str, default='-',
                       help='Input file (use "-" or omit for stdin)')
    parser.add_argument('--output', '-o', type=str, default='-',
                       help='Output file (use "-" or omit for stdout, default: stdout)')
    parser.add_argument('--format', choices=['conllu', 'teitok', 'plain', 'text', 'raw', 'auto'],
                       help='Input format (auto-detected from file extension if not specified)')
    parser.add_argument('--output-format', choices=['conllu', 'plain', 'text', 'plain-tagged'],
                       help='Output format (defaults to input format or conllu)')
    parser.add_argument('--segment', action='store_true',
                       help='Segment raw text into sentences')
    parser.add_argument('--tokenize', action='store_true',
                       help='Tokenize sentences into words')
    parser.add_argument('--model', type=Path, help='Path to trained model')
    parser.add_argument('--bert-model', default='bert-base-multilingual-cased',
                       help='BERT base model if no trained model')
    parser.add_argument('--vocab', type=Path,
                       help='Vocabulary file (JSON)')
    parser.add_argument('--vocab-priority', action='store_true',
                       help='Give vocabulary priority over model predictions')
    parser.add_argument('--respect-existing', action='store_true', default=True,
                       help='Respect existing annotations (default: True)')
    parser.add_argument('--no-respect-existing', dest='respect_existing', action='store_false',
                       help='Ignore existing annotations')
    parser.add_argument('--parse', action='store_true',
                       help='Run parsing (predict head and deprel)')
    parser.add_argument('--tag-only', action='store_true',
                       help='Only tag (UPOS/XPOS/FEATS), skip parsing')
    parser.add_argument('--parse-only', action='store_true',
                       help='Only parse (assumes tags already exist)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize orthographic variants')
    parser.add_argument('--normalization-attr', default='reg',
                       help='TEITOK attribute name for normalization')
    parser.add_argument('--expan', default='expan',
                       help='TEITOK attribute name or CoNLL-U MISC key for expansion')
    parser.add_argument('--xpos-attr', default='xpos',
                       help='TEITOK attribute name(s) for XPOS')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--lemma-anchor', choices=['reg', 'form', 'both'], default='both',
                       help='Anchor for learning inflection suffixes')
    
    args = parser.parse_args()
    
    # Handle stdin/stdout
    input_path = None
    output_path = None
    use_stdin = (args.input == '-' or args.input is None or not args.input)
    use_stdout = (args.output == '-' or args.output is None or not args.output)
    
    # Check if input is from terminal or pipe
    if use_stdin:
        if sys.stdin.isatty():
            print("Error: No input provided and stdin is a terminal", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        # Create temporary file from stdin
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write(sys.stdin.read())
            input_path = Path(tmp.name)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
    
    if not use_stdout:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure expansion key
    set_conllu_expansion_key(args.expan)
    load_teitok_xml._xpos_attr = args.xpos_attr
    load_teitok_xml._expan_attr = args.expan
    
    # Determine parse/tag settings
    parse_enabled = args.parse
    if args.tag_only:
        parse_enabled = False
    if args.parse_only:
        parse_enabled = True
    
    # Build config
    config = FlexiPipeConfig(
        bert_model=args.bert_model,
        respect_existing=args.respect_existing,
        parse=parse_enabled,
        tag_only=args.tag_only,
        parse_only=args.parse_only,
        vocab_priority=args.vocab_priority,
        normalize=args.normalize,
        normalization_attr=args.normalization_attr,
        debug=args.debug,
        lemma_anchor=args.lemma_anchor,
    )
    
    # Load vocabulary
    vocab = {}
    transition_probs = None
    if args.vocab:
        with open(args.vocab, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
            vocab = vocab_data.get('vocab', {})
            transition_probs = vocab_data.get('transitions', None)
        else:
            vocab = vocab_data
    
    # Create tagger
    tagger = FlexiPipeTagger(config, vocab, model_path=args.model if args.model else None, transition_probs=transition_probs)
    if args.model:
        tagger.load_model(args.model)
    
    # Auto-detect format
    input_format = args.format
    if not input_format:
        if use_stdin:
            input_format = 'plain'
        else:
            input_ext = input_path.suffix.lower()
            if input_ext == '.xml':
                input_format = 'teitok'
            elif input_ext in ('.conllu', '.conll'):
                input_format = 'conllu'
            else:
                input_format = 'plain'
    
    # Determine output format
    output_format = args.output_format or input_format
    if not args.output_format and input_format in ('plain', 'raw'):
        output_format = 'conllu'
    
    # Auto-enable segment/tokenize for 'raw' format
    segment = args.segment or (input_format == 'raw')
    tokenize = args.tokenize or (input_format == 'raw')
    
    # Tag
    tagged = tagger.tag(input_path, None, input_format, segment=segment, tokenize=tokenize)
    
    # Write output
    if use_stdout:
        tagger.write_output(tagged, None, output_format)
    else:
        tagger.write_output(tagged, output_path, output_format)
    
    # Clean up temporary file if created
    if use_stdin and input_path.exists():
        input_path.unlink()

if __name__ == '__main__':
    main()

