"""
Main entry point for FlexiPipe.
Dispatches to appropriate CLI module based on command.
"""

import sys
from pathlib import Path

# Add flexipipe to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main entry point that dispatches to CLI modules."""
    if len(sys.argv) < 2:
        print("Usage: flexipipe <command> [options]")
        print("\nCommands:")
        print("  train       - Train a model")
        print("  tag         - Tag text (supports stdin/stdout)")
        print("  analyze     - Analyze vocabulary and suffixes")
        print("  create-vocab - Create vocabulary from corpus")
        print("  normalizer  - Standalone BERT-based normalizer (optional)")
        sys.exit(1)
    
    command = sys.argv[1]
    # Remove command from argv so the CLI module sees only its arguments
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'train':
        from flexipipe.cli.train import main as train_main
        train_main()
    elif command == 'tag':
        from flexipipe.cli.tag import main as tag_main
        tag_main()
    elif command == 'analyze':
        from flexipipe.cli.analyze import main as analyze_main
        analyze_main()
    elif command == 'create-vocab':
        from flexipipe.cli.create_vocab import main as create_vocab_main
        create_vocab_main()
    elif command == 'normalizer':
        from flexipipe.cli.normalizer import main as normalizer_main
        normalizer_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Available commands: train, tag, analyze, create-vocab, normalizer", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

