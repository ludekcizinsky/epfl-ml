import numpy as np
import random

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """

    # Start with computing the subgradient for each datapoint
    e = y - tx @ w # (N,)
    n = e.shape[0]
    abs_subgrad = np.zeros((n, 2))
    
    for i in range(n):
        if e[i] > 0:
            abs_subgrad[i] = -tx[i] # -X
        elif e[i] < 0:
            abs_subgrad[i] = tx[i] # X
        else:
            print(">>>>>>>> Using subgradient!")
            # For each weight, sample from the range of [-Xi, Xi]
            m = tx[i].shape[0]
            subgradients = []
            for j in range(m):
                x = tx[i, j]
                g = random.uniform(-x, x)
                subgradients.append(g)
            abs_subgrad[i] = np.array(subgradients)

    # Now, we take the average
    return np.mean(abs_subgrad, axis=0)