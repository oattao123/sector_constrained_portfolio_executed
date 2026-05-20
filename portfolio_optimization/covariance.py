import torch
import numpy as np
import pandas as pd

def get_best_device():
    """Auto-detects the best available PyTorch device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_tensor(data, device=None, dtype=torch.float32):
    """Converts numpy array, pandas dataframe or list to PyTorch tensor on specified device."""
    if device is None:
        device = get_best_device()
        
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return torch.tensor(data.values, dtype=dtype, device=device)
    else:
        return torch.tensor(np.array(data), dtype=dtype, device=device)

def ledoit_wolf_covariance_gpu_dynamic(X):
    """
    Computes Ledoit-Wolf Shrinkage Covariance dynamically using PyTorch.
    
    Parameters:
    -----------
    X : torch.Tensor
        Centered/uncentered log returns tensor of shape (n_samples, n_features)
        
    Returns:
    --------
    shrunk_cov : torch.Tensor
        Shrunk covariance matrix (n_features, n_features)
    sample_cov : torch.Tensor
        Sample covariance matrix (n_features, n_features)
    delta : float
        Optimal shrinkage parameter delta
    """
    n_samples, n_features = X.shape
    
    # 1. Centering Data
    X_centered = X - torch.mean(X, dim=0)
    
    # 2. Sample Covariance (S)
    sample_cov = torch.matmul(X_centered.T, X_centered) / n_samples
    
    # 3. Target Covariance (T)
    mean_var = torch.trace(sample_cov) / n_features
    target_cov = mean_var * torch.eye(n_features, device=X.device)
    
    # 4. Dynamic Delta (Optimal Shrinkage)
    d_squared = torch.sum((sample_cov - target_cov) ** 2)
    
    X_squared = X_centered ** 2
    b_squared = torch.sum((torch.matmul(X_squared.T, X_squared) / n_samples) - (sample_cov ** 2)) / n_samples
    
    delta = b_squared / d_squared
    delta = torch.clamp(delta, min=0.0, max=1.0)
    
    # 5. Shrunk Covariance
    shrunk_cov = (1 - delta) * sample_cov + delta * target_cov
    
    return shrunk_cov, sample_cov, delta.item()

def compute_correlation_matrix(cov_matrix):
    """
    Computes correlation matrix from a covariance matrix.
    
    Parameters:
    -----------
    cov_matrix : torch.Tensor
        Covariance matrix (n_features, n_features)
        
    Returns:
    --------
    corr_matrix : torch.Tensor
        Correlation matrix (n_features, n_features)
    """
    vols = torch.sqrt(torch.diag(cov_matrix))
    outer_vols = torch.outer(vols, vols)
    # Avoid division by zero
    outer_vols = torch.clamp(outer_vols, min=1e-8)
    corr_matrix = cov_matrix / outer_vols
    
    # Clamp to avoid floating point errors out of boundary [-1, 1]
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
    return corr_matrix
