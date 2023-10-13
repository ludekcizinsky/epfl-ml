# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w, name="MSE"):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = tx.shape[0]
    if name == "MSE":
        return np.sum((y - tx @ w)**2)/2*N
    elif name == "MAE":
        return np.sum(np.absolute((y - tx @ w)))/2*N
    else:
        raise NotImplementedError 