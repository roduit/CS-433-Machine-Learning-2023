# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    e = y - tx @ w
    loss = 1/(2*len(y)) *np.linalg.norm(e)**2
    return loss
    # ***************************************************
