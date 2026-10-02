import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # NumPy random
    torch.manual_seed(seed)                # PyTorch CPU seed
    torch.cuda.manual_seed(seed)           # PyTorch GPU seed (single GPU)
    torch.cuda.manual_seed_all(seed)       # PyTorch GPU seed (all GPUs if multi-GPU)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False