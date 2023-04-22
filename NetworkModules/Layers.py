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
        self.weights = np.random.randn(input_size, output_size)* np.sqrt(1 / input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights.T, input) + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        
        M  = output_gradient.shape[1]

        input_gradient = np.dot(self.weights, output_gradient) / M

        weight_update = np.dot(output_gradient, self.input.T).T / M   # Why is the .T necessary
        bias_update   = np.sum(output_gradient, axis = 1, keepdims = True) / M
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
    
    def backward(self, output_gradient, learning_rate = None):
        return self.activation_gradient(self.input) * output_gradient  # element wise product
    


# %%
if __name__ == "__main__":
    from activations import *
    from loss_functions import *

    linear_layer_1 = FullyConnectedLinearLayer(input_size=20, output_size=10)
    activation_layer_1 = ActivationLayer(relU, drelU)
    linear_layer_2 = FullyConnectedLinearLayer(input_size=10, output_size=5)
    activation_layer_2 = ActivationLayer(softmax, dsoftmax)

    X_input = np.random.rand(20, 200)
    Y_truth = np.random.rand(5, 200)


    print('Forward Prop')
    linear_layer_1_output = linear_layer_1.forward(X_input)
    print(f'Linear output 1 shape {linear_layer_1_output.shape}')

    activation_layer_1_output = activation_layer_1.forward(linear_layer_1_output)
    print(f'Activation output 1 shape {activation_layer_1_output.shape}')

    linear_layer_2_output = linear_layer_2.forward(activation_layer_1_output)
    print(f'Linear output 1 shape {linear_layer_1_output.shape}')

    activation_layer_2_output = activation_layer_2.forward(linear_layer_2_output)
    print(f'Activation output 2 shape {activation_layer_2_output.shape}')


    print('Back Prop')
    loss_gradient = dcross_entropy_loss(activation_layer_2_output, Y_truth)
    print(f'Loss gradient shape {loss_gradient.shape}')

    activation_2_grad = activation_layer_2.backward(loss_gradient)
    print(f'Activation 2 gradient shape {activation_2_grad.shape}')

    linear_2_grad = linear_layer_2.backward(activation_2_grad, 0.005)
    print(f'Linear 2 gradient shape {linear_2_grad.shape}')

    activation_1_grad = activation_layer_1.backward(linear_2_grad)
    print(f'Activation 1 gradient shape {activation_1_grad.shape}')

    linear_1_grad = linear_layer_1.backward(activation_1_grad, 0.005)
    print(f'Linear 1 gradient shape {linear_1_grad.shape}')
# %%
