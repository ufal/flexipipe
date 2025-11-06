"""
Utility functions for FlexiPipe, including device detection.
"""

try:
    import torch
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_IMPORT_ERROR = None
    
    def get_device():
        """Detect and return the best available device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_IMPORT_ERROR = e
    
    def get_device():
        """Fallback device detection when torch is not available."""
        return None

