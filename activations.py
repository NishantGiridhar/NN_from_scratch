# %%

import autograd.numpy as np
from autograd import grad

# tanh and derivative
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

# relU and derivative
def relU(x):
    return np.maximum(0, x)

def drelU(x):
    a = np.zeros(x.shape,dtype=np.float32)
    return np.maximum(a, x)

# Sigmoid and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return np.exp(-x)/(np.exp(-x)+1)**2

# softmax and derivative
def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def dsoftmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps/np.sum(exps)*(1-exps/np.sum(exps))

# %%
