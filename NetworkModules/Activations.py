# %%

import numpy as np


# tanh and derivative
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

# relU and derivative
def relU(x):
    return np.maximum(0, x)

def drelU(x):
    return np.maximum(0, 1)

# Sigmoid and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return np.exp(-x)/(np.exp(-x)+1)**2

# softmax and derivative
def softmax(x):
    shiftx = x - np.max(x, axis = 0, keepdims=True)
    exps = np.exp(shiftx)
    den = np.sum(exps, axis = 0, keepdims=True)
    return exps / den

def dsoftmax(x):
    # shiftx = x - np.max(x)
    # exps = np.exp(shiftx)
    return 1#exps/np.sum(exps, axis = 0)*(1-exps/np.sum(exps, axis = 0))

# %%
