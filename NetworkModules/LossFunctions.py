# %%

import numpy as np

def mse_loss(y, y_truth):
    return np.mean(np.power(y_truth-y, 2))

def dmse_loss(y, y_truth):
    return - 2 * (y_truth-y) / y_truth.size


def cross_entropy_loss(y, y_truth):
    N = y.shape[0]
    M = y.shape[1]
    ce = - np.sum(y_truth * np.log(y)) / N / M
    return ce

def dcross_entropy_loss(y, y_truth):
    return (y - y_truth)
# %%
