#!/bin/bash
# Script to train FlexiPipe on German-GSD treebank

# Check if transformers is available
python3 -c "import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: transformers library not installed"
    echo "Please install dependencies with:"
    echo "  pip install transformers torch datasets scikit-learn accelerate"
    echo ""
    echo "Or if using a virtual environment:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install transformers torch datasets scikit-learn accelerate"
    echo ""
    echo "Note: accelerate>=0.26.0 is required for training"
    exit 1
fi

# Training parameters
# Optimized for SOTA performance:
# - MLP heads (2-layer with GELU activation)
# - Loss weighting (UPOS 2.0x, XPOS 1.5x, FEATS 1.0x)
# - Learning rate warmup (500 steps) + cosine decay
# - Removed UPOS context tokens (were hurting performance)
# - Increased batch size and epochs for better convergence
DATA_DIR="/Volumes/Data2/Git/UD/ud_treebanks/UD_German-GSD"
BERT_MODEL="bert-base-german-cased"
OUTPUT_DIR="models/flexipipe-german-gsd"
BATCH_SIZE=16  # Will be auto-reduced to 2 for parser training (effective batch maintained via gradient accumulation)
GRADIENT_ACCUMULATION_STEPS=2  # Will be auto-increased to 8 for parser training to maintain effective batch size
LEARNING_RATE=2e-5
NUM_EPOCHS=5  # Increased from 3 - BERT fine-tuning often needs more epochs

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Training FlexiPipe on German-GSD treebank..."
echo "  Data directory: $DATA_DIR"
echo "  (Looking for *-ud-train.conllu and *-ud-dev.conllu)"
echo "  BERT model: $BERT_MODEL"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE (with warmup + cosine decay)"
echo "  Epochs: $NUM_EPOCHS"
echo ""
echo "Training components (default: all enabled):"
echo "  - Tokenizer: enabled"
echo "  - Tagger (UPOS/XPOS/FEATS): enabled"
echo "  - Lemmatizer (LEMMA): enabled"
echo "  - Parser (HEAD/DEPREL): enabled"
echo "  (Use --no-tokenizer, --no-tagger, --no-lemmatizer, or --no-parser to disable components)"
echo ""
echo "Note: Parser training is memory-intensive. Batch size will be automatically"
echo "  reduced if parser is enabled (maintaining effective batch via gradient accumulation)."
echo ""
echo "Architecture improvements:"
echo "  - MLP classification heads (2-layer with GELU)"
echo "  - Weighted loss (UPOS 2.0x, XPOS 1.5x, FEATS 1.0x)"
echo "  - Learning rate warmup (500 steps) + cosine decay"
echo "  - Removed UPOS context tokens (better performance)"
echo ""

# Run training
python3 flexipipe.py train \
    --data-dir "$DATA_DIR" \
    --bert-model "$BERT_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --num-epochs "$NUM_EPOCHS"

if [ $? -eq 0 ]; then
    echo ""
    echo "Training complete! Model saved to: $OUTPUT_DIR"
    echo ""
    echo "To use the model for tagging:"
    echo "  python3 flexipipe.py tag input.conllu --output output.conllu --model $OUTPUT_DIR"
    echo ""
    echo "To check accuracy on test set:"
    echo "  bash check_flexipipe_accuracy.sh"
    echo ""
    echo "Expected performance (with improvements):"
    echo "  - UPOS: ~94-96% (target: 97% like UDPIPE2)"
    echo "  - XPOS: ~94-96%"
    echo "  - UFeats: ~85-90%"
    echo "  - AllTags: ~85-90%"
    echo "  - UAS: ~85-90% (dependency parsing)"
    echo "  - LAS: ~80-85% (labeled dependency parsing)"
else
    echo ""
    echo "Training failed. Check error messages above."
    exit 1
fi

