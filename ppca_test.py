"""
Implement PPCA using optimization to solve for v and sigma^2 instead of matrix decomposition (eigenvalue formula).


"""

import numpy as np
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA (GPU) available? {torch.cuda.is_available()}")

# Create a random vector v
v = torch.randn(3, 1)
print("Initial v vector:\n", v)