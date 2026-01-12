# fastText Backend Notes

## Why fastText Uses Different Hyperparameters Than Deep Neural Networks

### Learning Rate (0.1-1.0 vs 1e-5)

**fastText uses much higher learning rates** (typically 0.1-0.7, default 0.5) compared to deep neural networks (typically 1e-5 to 1e-3) because:

1. **Simpler Architecture**: fastText is based on word2vec/skip-gram, which uses simple linear transformations and shallow networks, not deep multi-layer networks.

2. **Stochastic Gradient Descent**: fastText processes one example at a time with online SGD, which is more stable with higher learning rates than batch-based training.

3. **Different Update Rules**: The gradient updates in word2vec-style models are simpler and can handle larger step sizes without diverging.

4. **Vocabulary Size**: With large vocabularies, each word vector gets updated infrequently, so higher learning rates help ensure meaningful updates.

**When to use lower learning rates:**
- Fine-tuning on small datasets
- When you see training instability
- For very large models (dim > 300)

### No Batch Size

**fastText processes examples one at a time** (online learning), not in batches. This is by design:

1. **Memory Efficiency**: Processing one example at a time uses minimal memory, making it suitable for very large datasets.

2. **Simplicity**: No need to manage batch buffers, padding, or batching logic.

3. **Online Learning**: The model updates immediately after each example, which can be beneficial for streaming data.

4. **Efficiency**: For the simple fastText architecture, batching doesn't provide significant speedup.

**Implication**: You cannot control batch size. The model processes the entire training file sequentially, one line at a time.

### No Weight Decay

**fastText doesn't support explicit weight decay/L2 regularization**, but uses alternative regularization methods:

1. **minCount**: Filters out words that appear fewer than `minCount` times. This acts as regularization by preventing the model from overfitting to rare words.

2. **wordNgrams**: Using word n-grams (default: 3) helps capture context and improves generalization.

3. **Early Stopping**: Monitor dev set performance and stop training when it plateaus.

4. **Lower Learning Rates**: For fine-tuning, use lower learning rates (0.01-0.1) to prevent overfitting.

5. **Character n-grams (minn/maxn)**: Help with out-of-vocabulary words and provide additional regularization.

**Why no weight decay?**
- The fastText library doesn't expose this parameter
- The architecture is simple enough that other regularization methods are sufficient
- The vocabulary filtering (minCount) provides effective regularization

## Recommendations

### For Best Accuracy:
- Use `--finetune accuracy` to search hyperparameters
- Try learning rates: 0.1, 0.3, 0.5, 0.7
- Increase epochs: 40-50
- Increase dimensions: 200-300
- Use warmup: `--lr-warmup 5 --lr-warmup-ratio 0.1`

### For Faster Training:
- Use `--finetune speed`
- Lower epochs: 20-30
- Lower dimensions: 100-200
- Standard learning rate: 0.5

### For Small Datasets:
- Lower learning rate: 0.1-0.3
- Increase minCount: 2-3 (filters rare words)
- More epochs: 40-50
- Use warmup to stabilize training

