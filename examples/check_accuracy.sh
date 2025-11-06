#!/bin/bash
# Script to check accuracy of trained FlexiPipe model against UD treebank test set

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment: venv"
elif [ -d "eflomal-env" ]; then
    source eflomal-env/bin/activate
    echo "Activated virtual environment: eflomal-env"
fi

MODEL_DIR="models/flexipipe-german-gsd"
TEST_FILE="/Volumes/Data2/Git/UD/ud_treebanks/UD_German-GSD/de_gsd-ud-test.conllu"
PRED_FILE="tmp/test_predictions.conllu"

echo "Checking accuracy of FlexiPipe model: $MODEL_DIR"
echo "Test file: $TEST_FILE"
echo ""

# Step 1: Tag the test file
echo "Step 1: Tagging test file..."
echo "Note: Using --no-respect-existing to force model predictions (not copying gold annotations)"
echo "Note: Adding --parse flag if model supports parsing (check label_mappings.json for deprel_labels)"

# Check if model has parser (deprel_labels in label_mappings.json)
HAS_PARSER=false
if [ -f "$MODEL_DIR/label_mappings.json" ]; then
    # Check if deprel_labels exists and is a non-empty array
    DEPREL_COUNT=$(python3 -c "import json; d=json.load(open('$MODEL_DIR/label_mappings.json')); print(len(d.get('deprel_labels', [])))" 2>/dev/null)
    if [ "$DEPREL_COUNT" -gt 0 ] 2>/dev/null; then
        HAS_PARSER=true
    fi
fi

# Build tag command
TAG_CMD="python3 flexipipe.py tag \"$TEST_FILE\" --output \"$PRED_FILE\" --model \"$MODEL_DIR\" --format conllu --no-respect-existing"

if [ "$HAS_PARSER" = true ]; then
    echo "Model appears to have parser support, adding --parse flag"
    TAG_CMD="$TAG_CMD --parse"
else
    echo "Model does not appear to have parser support, skipping --parse"
fi

eval $TAG_CMD

if [ $? -ne 0 ]; then
    echo "Error: Tagging failed"
    exit 1
fi

echo ""
echo "Step 2: Calculating accuracy..."
python3 flexipipe.py calculate-accuracy \
    "$TEST_FILE" \
    "$PRED_FILE" \
    --format conllu

if [ $? -eq 0 ]; then
    echo ""
    echo "Accuracy check complete!"
    echo "Predictions saved to: $PRED_FILE"
else
    echo "Error: Accuracy calculation failed"
    exit 1
fi

