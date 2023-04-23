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
        self.weights = np.random.randn(input_size, output_size).T* np.sqrt(1 / input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, input) + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate, _reg = False, _reg_lam = 0.2):
        
        M  = output_gradient.shape[1]
        # % \\print(f'Shape of M is {M}')
        # input()

        input_gradient = np.dot(self.weights.T, output_gradient) 

        weight_update = np.dot(output_gradient, self.input.T) / M   # Why is the .T necessary
        bias_update   = np.sum(output_gradient, axis = 1, keepdims = True) / M
        if _reg == False:
            self.weights += - learning_rate * weight_update
        else:
            self.weights += - learning_rate * (weight_update + _reg_lam / M *self.weights)
        self.biases += - learning_rate * bias_update

        return input_gradient


class ActivationLayer(GenericLayer):
    def __init__(self, activation_function, activation_gradient):
        self.activation             = activation_function
        self.activation_gradient    = activation_gradient
        self.keep_prob = 1

    def forward(self, input):
        self.input = input
        self.output = self.activation(input) 
        return self.output
    
    def backward(self, output_gradient, learning_rate = None, _reg = None, _reg_lam = None):
        return np.multiply(np.int64(self.activation_gradient(self.input)>0) , output_gradient)  # element wise product
    
    def _add_dropout(self, keep_prob):
        self.keep_prob = keep_prob
        
    


# %%
if __name__ == "__main__":
    from Activations import *
    from LossFunctions import *

    linear_layer_1 = FullyConnectedLinearLayer(input_size=25, output_size=15)
    activation_layer_1 = ActivationLayer(relU, drelU)
    linear_layer_2 = FullyConnectedLinearLayer(input_size=15, output_size=10)
    activation_layer_2 = ActivationLayer(softmax, dsoftmax)

    X_input = np.random.rand(25, 200)
    Y_truth = np.random.rand(10, 200)


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
