import numpy as np

class network:
    def __init__(self):
        self.layers = []
        self.loss   = None
        self.loss_derivative = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_function, loss_derivative):
        self.loss = loss_function
        self.loss_derivative = loss_derivative

    def load_data(self, x_train = None, y_train = None, x_test = None, y_test = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test  = x_test
        self.y_test  = y_test

    def _batch_gradient_descent(self, x_train, y_train, learning_rate):
        return None