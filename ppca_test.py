"""
Implement PPCA using optimization to solve for v and sigma^2 instead of matrix decomposition (eigenvalue formula).
Goal: deconstruct S to get v and sigma^2 via gradient descent

"""

import numpy as np
import torch

# Set up dimensions: 200 samples, 3 dimensions/features
N, P = 200, 3

# Set true signal
true_v = torch.tensor([[1.0], [2.0], [3.0]]) 
true_sigma_sq = 0.5

## Create data matrix (200 samples, 3 features)
# data = torch.randn(N, P) # true random data
latent_x = torch.randn(N, 1) # signal along one dimension
noise = torch.randn(N, P) * np.sqrt(true_sigma_sq)
# map 1D signal to 3D using true_v
data = latent_x @ true_v.T + noise
print(f"Shape of data: {data.shape}")
print(f"Raw column means are {data.mean(dim=0)}")
# Center data
data = data - data.mean(dim=0) # dim = 0, collapse along rows

# Calculate sample covariance matrix from data
# S = Cov(y_i) = vv^T + sigma^2 * Ib
S = (1/N)*(data.T @ data)
print(f"Shape of covariance matrix: {S.shape}")
print(f"Covariance matrix: {S}")

## Theoretical Solution via Eigenvalues
# Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix
eigvalues, eigvecs = torch.linalg.eigh(S)
print(f"Shape of eigenvalues: {eigvalues.shape}")
print(f"Eigenvalues: {eigvalues}")
print(f"Shape of eigenvectors: {eigvecs.shape}")
print(f"Eigenvectors: {eigvecs}")

# Take largest eigenvalue (amount of variance explained) - signal
lambda_1 = eigvalues[-1]
u_1 = eigvecs[:, -1:]
print(f"Total variance in direction: {lambda_1}")
print(f"Direction: {u_1}")

# Calculate theoretical sigma^2
theoretical_sigma_sq = (eigvalues[0] + eigvalues[1]) / 2
print(f"Theoretical sigma^2: {theoretical_sigma_sq}")

# Calculate theoretical signal
theoretical_v = u_1 * torch.sqrt(lambda_1 - theoretical_sigma_sq)
print(f"Theoretical v: {theoretical_v}")