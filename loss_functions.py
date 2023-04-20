# %%

import numpy as np

def mse_loss(y, y_truth):
    return np.mean(np.power(y_truth-y, 2))

def dmse_loss(y, y_truth):
    return 2 * (y_truth-y) / y_truth.size


def cross_entropy_loss(y, y_truth):
    N = y.shape[0]
    ce = - np.sum(y_truth * np.log(y)) / N
    return ce

def dcross_entropy_loss(y, y_truth):
    N = y.shape[0]
    return - np.sum(np.divide(y_truth, y)) / N
# %%
