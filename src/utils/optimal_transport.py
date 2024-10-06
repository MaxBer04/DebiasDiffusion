"""
Optimal Transport Module for DebiasDiffusion

This module provides a function to solve the optimal transport problem
for fairness considerations in the DebiasDiffusion project. It uses the
SciPy library to compute the optimal assignment between predicted probabilities
and target distributions.

Usage:
    from src.utils.optimal_transport import solve_optimal_transport

    assignments = solve_optimal_transport(probs, targets)

Note:
    This module requires the NumPy and SciPy libraries to be installed.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple

def solve_optimal_transport(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Solve the optimal transport problem between predicted probabilities and target distributions.

    Args:
        probs (np.ndarray): Predicted probability distributions of shape (batch_size, num_classes).
        targets (np.ndarray): Target class indices of shape (batch_size,).

    Returns:
        np.ndarray: Optimal assignments of shape (batch_size,).

    Note:
        This function uses the Hungarian algorithm to solve the assignment problem,
        which has a time complexity of O(n^3) where n is the batch size.
    """
    batch_size, num_classes = probs.shape
    
    # Convert targets to one-hot vectors
    target_one_hot = np.eye(num_classes)[targets]
    
    # Compute cost matrix
    cost_matrix = np.sum((probs[:, np.newaxis, :] - target_one_hot[np.newaxis, :, :]) ** 2, axis=2)
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create the assignment array
    assignments = np.zeros(batch_size, dtype=int)
    assignments[row_ind] = targets[col_ind]
    
    return assignments