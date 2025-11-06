# Multi-Platform Support: CUDA, MPS, and CPU

FlexiPipe automatically detects and optimizes for the available hardware platform.

## Device Detection Priority

The code detects devices in this order:
1. **MPS** (Apple Silicon GPU) - if available
2. **CUDA** (NVIDIA GPU) - if available  
3. **CPU** - fallback

Device detection happens automatically - no configuration needed.

## Platform-Specific Optimizations

### CUDA (NVIDIA GPUs)

**Training Optimizations:**
- ✅ **fp16 mixed precision**: Enabled automatically (faster, less memory)
- ✅ **Larger batch sizes**: Can use batch_size=4 for parser training (vs 1 for MPS)
- ✅ **DataLoader workers**: Uses 2 workers for faster data loading
- ✅ **Pin memory**: Enabled for faster GPU transfer

**Performance:**
- Fastest training speed
- Can handle larger models and batches
- Best for production training on GPU clusters

### MPS (Apple Silicon - M1/M2/M3)

**Training Optimizations:**
- ✅ **Gradient checkpointing**: Enabled automatically (trades compute for memory)
- ✅ **Small batch sizes**: Uses batch_size=1 for parser training to avoid OOM
- ✅ **Reduced max_length**: Caps at 128 for parser training (vs 512 default)
- ✅ **Cache clearing**: Periodic MPS cache clearing to prevent memory accumulation
- ✅ **Frequent checkpoints**: Saves every 250 steps (vs 500) to avoid losing progress
- ❌ **fp16 disabled**: MPS doesn't support fp16 well
- ❌ **No data workers**: Set to 0 to avoid multiprocessing issues

**Memory Management:**
- Very aggressive memory optimization due to unified memory architecture
- May be slower than CUDA but more stable for long training runs
- Best for development and smaller models

### CPU

**Training Optimizations:**
- Uses batch_size=2 for parser training
- No GPU-specific optimizations
- Slower but always available

## Code Changes Made

### Device Detection
```python
def get_device():
    """Detect and return the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

### Platform-Specific Training Settings

**CUDA:**
- `fp16=True` - Mixed precision training
- `dataloader_num_workers=2` - Parallel data loading
- `dataloader_pin_memory=True` - Faster GPU transfer
- `batch_size=4` for parser (if original >= 16)

**MPS:**
- `fp16=False` - Not well supported
- `dataloader_num_workers=0` - Avoid multiprocessing issues
- `dataloader_pin_memory=False` - Suppress warnings
- `batch_size=1` for parser - Maximum memory safety
- `max_length=128` for parser - Reduce arc score memory
- Gradient checkpointing enabled
- Cache clearing callback every 50 steps

## Installation

### For CUDA Cluster
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers datasets scikit-learn accelerate
```

### For MPS (Apple Silicon)
```bash
# Install PyTorch (MPS support included by default)
pip install torch torchvision torchaudio

# Install other dependencies
pip install transformers datasets scikit-learn accelerate
```

### For CPU
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install transformers datasets scikit-learn accelerate
```

## Usage

The same commands work on all platforms - device is auto-detected:

```bash
# Training
python flexipipe.py train --data-dir /path/to/treebank --output-dir models/flexipipe

# Tagging
python flexipipe.py tag input.conllu --model models/flexipipe --output output.conllu
```

The code will automatically:
- Detect the available device
- Apply platform-specific optimizations
- Print which device is being used

## Troubleshooting

### MPS Out of Memory
If you get OOM errors on MPS:
- The code already uses batch_size=1 and max_length=128
- Try disabling parser training: `--no-parser`
- Or train on CPU: Set `CUDA_VISIBLE_DEVICES=""` (though MPS doesn't use this)

### CUDA Out of Memory
If you get OOM errors on CUDA:
- Reduce batch size: `--batch-size 2`
- Reduce max length: Modify code to use smaller max_length
- Enable gradient checkpointing (currently only for MPS)

### Performance Comparison
- **CUDA**: Fastest, best for production
- **MPS**: Good for development, may be slower but more stable
- **CPU**: Slowest, but always available

## Testing Multi-Platform Support

To test on different platforms, the code will automatically adapt. No code changes needed when moving between platforms.

