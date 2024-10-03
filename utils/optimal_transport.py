import numpy as np
from scipy.optimize import linear_sum_assignment

def solve_optimal_transport(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
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