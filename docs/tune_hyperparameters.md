# FlexiPipe Hyperparameter Tuning Guide

## Key Hyperparameters to Tune

### 1. Learning Rate (Most Important)
- **Current**: `2e-5`
- **Range to try**: `1e-5`, `2e-5`, `3e-5`, `5e-5`
- **Impact**: Critical for BERT fine-tuning. Too high = unstable training, too low = slow convergence
- **Recommendation**: Start with `2e-5`, try `3e-5` if training is stable

### 2. Batch Size
- **Current**: `16` (with gradient accumulation = 2, effective batch = 32)
- **Range to try**: `8`, `12`, `16`, `24` (if memory allows)
- **Impact**: Larger batches = more stable gradients but higher memory
- **Memory constraint**: MPS (Apple Silicon) has limited memory, so 16 is a good default
- **Effective batch**: Keep `batch_size * gradient_accumulation_steps` around 32-64

### 3. Gradient Accumulation Steps
- **Current**: `2`
- **Range to try**: `1`, `2`, `4`, `8`
- **Impact**: Simulates larger batch size without memory cost
- **Formula**: Effective batch = `batch_size * gradient_accumulation_steps`

### 4. Number of Epochs
- **Current**: `5`
- **Impact**: Early stopping should handle this, but you can adjust
- **Recommendation**: Keep at 5, let early stopping decide when to stop

### 5. Warmup Steps
- **Current**: `500` steps or `0.1` ratio
- **Range to try**: `250`, `500`, `1000` steps or `0.05`, `0.1`, `0.15` ratio
- **Impact**: Helps stabilize training at the start
- **Recommendation**: `500` steps or `10%` of training steps is usually good

### 6. Learning Rate Scheduler
- **Current**: `cosine`
- **Alternatives**: `linear`, `polynomial`, `constant_with_warmup`
- **Impact**: Cosine decay is usually best for BERT fine-tuning

### 7. Weight Decay
- **Current**: `0.01`
- **Range to try**: `0.0`, `0.01`, `0.05`, `0.1`
- **Impact**: Regularization to prevent overfitting
- **Recommendation**: `0.01` is standard for BERT

### 8. MLP Hidden Size (Architecture)
- **Current**: `hidden_size // 2` (384 for BERT-base)
- **Range to try**: `hidden_size // 4`, `hidden_size // 2`, `hidden_size * 2 // 3`
- **Impact**: Larger = more capacity but slower and more memory
- **Location**: In `MultiTaskFlexiPipe.__init__`, `mlp_hidden = hidden_size // 2`

### 9. Dropout Rate
- **Current**: `0.1` in MLP heads
- **Range to try**: `0.0`, `0.1`, `0.2`, `0.3`
- **Impact**: Higher dropout = more regularization but may hurt performance
- **Recommendation**: `0.1` is standard

### 10. Loss Weights
- **Current**: UPOS `2.0x`, XPOS `1.5x`, FEATS `1.0x`
- **Range to try**: Adjust if UPOS is already good but XPOS/FEATS need improvement
- **Location**: In `MultiTaskFlexiPipe.forward()`

## Tuning Strategy

### Step 1: Learning Rate Search
1. Fix all other hyperparameters
2. Try learning rates: `1e-5`, `2e-5`, `3e-5`, `5e-5`
3. Train for 1-2 epochs and check validation loss
4. Choose the learning rate with lowest validation loss

### Step 2: Batch Size + Gradient Accumulation
1. Keep effective batch size around 32-64
2. Try combinations:
   - `batch_size=8, gradient_accumulation=4` (effective=32)
   - `batch_size=16, gradient_accumulation=2` (effective=32) ← current
   - `batch_size=16, gradient_accumulation=4` (effective=64)
   - `batch_size=24, gradient_accumulation=2` (effective=48) - if memory allows

### Step 3: Fine-tune Other Parameters
1. Adjust warmup steps if training is unstable at start
2. Adjust weight decay if overfitting (increase) or underfitting (decrease)
3. Adjust MLP hidden size if model capacity needs change

## Monitoring Performance

### Key Metrics to Watch:
1. **Training loss**: Should decrease smoothly
2. **Validation loss**: Should decrease, watch for overfitting (training loss decreases but validation loss plateaus/increases)
3. **UPOS accuracy**: Primary target (aim for 94-97%)
4. **XPOS accuracy**: Secondary target
5. **UFeats accuracy**: Usually lower, but should be >85%

### Early Stopping:
- Patience: 3 evaluation steps (1500 steps with `eval_steps=500`)
- Metric: `eval_loss` (lower is better)
- Model: Best model (lowest validation loss) is saved automatically

## Example Tuning Commands

```bash
# Try different learning rates
for lr in 1e-5 2e-5 3e-5 5e-5; do
    python3 flexipipe.py train \
        --data-dir /Volumes/Data2/Git/UD/ud_treebanks/UD_German-GSD \
        --output-dir models/flexipipe-lr-$lr \
        --learning-rate $lr \
        --batch-size 16 \
        --gradient-accumulation-steps 2 \
        --num-epochs 5
done

# Try different batch sizes (keeping effective batch ~32)
python3 flexipipe.py train \
    --data-dir /Volumes/Data2/Git/UD/ud_treebanks/UD_German-GSD \
    --output-dir models/flexipipe-bs8-ga4 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --num-epochs 5

python3 flexipipe.py train \
    --data-dir /Volumes/Data2/Git/UD/ud_treebanks/UD_German-GSD \
    --output-dir models/flexipipe-bs16-ga2 \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --num-epochs 5
```

## Expected Performance Targets

With current settings (MLP heads, loss weighting, warmup):
- **UPOS**: 94-97% (target: match UDPIPE2's 97%)
- **XPOS**: 94-96%
- **UFeats**: 85-92%
- **AllTags**: 85-90%

## Memory Constraints on MPS

If you hit OOM errors:
1. Reduce `batch_size` (try 8 or 12)
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_length` from 512 to 256 (if sentences are short)
4. Consider using CPU if memory is too constrained (slower but more stable)

## Quick Reference

| Hyperparameter | Current | Recommended Range | Priority |
|----------------|---------|-------------------|----------|
| Learning Rate | 2e-5 | 1e-5 to 5e-5 | ⭐⭐⭐ High |
| Batch Size | 16 | 8-24 | ⭐⭐ Medium |
| Gradient Accumulation | 2 | 1-4 | ⭐⭐ Medium |
| Warmup Steps | 500 | 250-1000 | ⭐ Low |
| Weight Decay | 0.01 | 0.0-0.1 | ⭐ Low |
| Epochs | 5 | 3-7 | ⭐ Low (early stopping) |

