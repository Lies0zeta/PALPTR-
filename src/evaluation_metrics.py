"""
Evaluation metrics for PTR experiments
@author: Lies Hadjadj
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def cosine_similarity_gradients(grad1: List[Tuple[np.ndarray]],
                              grad2: List[Tuple[np.ndarray]]) -> float:
    """
    Compute cosine similarity between two gradient vectors
    
    Args:
        grad1: First gradient list of tuples
        grad2: Second gradient list of tuples
        
    Returns:
        Cosine similarity between flattened gradients
    """
    # Flatten gradients
    grad1_flat = np.concatenate([g[0].flatten() for g in grad1])
    grad2_flat = np.concatenate([g[0].flatten() for g in grad2])

    # Compute cosine similarity
    return np.dot(grad1_flat, grad2_flat) / (
        np.linalg.norm(grad1_flat) * np.linalg.norm(grad2_flat)
    )


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate clustering accuracy using optimal label assignment
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
        
    Returns:
        Accuracy score in range [0,1]
        
    Raises:
        AssertionError: If input arrays have different sizes
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size, "Size mismatch between y_true and y_pred"

    # Create confusion matrix
    num_classes = max(y_pred.max(), y_true.max()) + 1
    weights = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(y_pred.size):
        weights[y_pred[i], y_true[i]] += 1

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(weights.max() - weights)

    # Calculate accuracy
    correct = sum(weights[i, j] for i, j in zip(row_ind, col_ind))
    return float(correct) / y_pred.size


# Aliases for commonly used sklearn metrics
nmi = normalized_mutual_info_score
ari = adjusted_rand_score
