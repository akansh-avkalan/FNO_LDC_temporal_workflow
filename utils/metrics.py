import numpy as np

def L2_norm(pred, target):
    """
    Computes the L2 (Euclidean) norm between prediction and target.
    Args:
        pred (np.ndarray): Predicted array of shape (N, H, W) or (B, C, H, W)
        target (np.ndarray): Ground truth array of same shape as pred
    Returns:
        float: L2 norm
    """
    return np.sqrt(np.mean((pred - target) ** 2))


def LInf_norm(pred, target):
    """
    Computes the L-infinity (max) norm between prediction and target.
    Args:
        pred (np.ndarray): Predicted array
        target (np.ndarray): Ground truth array
    Returns:
        float: L-infinity norm
    """
    return np.max(np.abs(pred - target))
