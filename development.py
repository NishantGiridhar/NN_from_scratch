# %%
import numpy as np

class GenericLayer:
    def __init__(self):
        self.input  = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_error, learning_rate):
        raise NotImplementedError
    

class FullyConnectedLinearLayer(GenericLayer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(output_size, 1) - 0.5

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights.T, input) + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate):

        input_gradient = np.dot(self.weights.T, output_gradient)

        weight_update = np.dot(output_gradient.T, self.input)
        bias_update   = output_gradient

        self.weights += - learning_rate * weight_update
        self.biases += - learning_rate * bias_update

        return input_gradient


class ActivationLayer(GenericLayer):
    def __init__(self, activation_function, activation_gradient):
        self.activation             = activation_function
        self.activation_gradient    = activation_gradient

    def forward(self, input):
        self.input = input
        self.output = self.activation(input) 
        return self.output
    
    def backward(self, output_gradient):
        return self.activation_gradient(self.input) * output_gradient  # element wise product
    


# %%
