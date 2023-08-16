# %%
import numpy as np
# This is a change
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
        
        # Gradient Descent Terms
        self.weights = np.random.randn(input_size, output_size).T* np.sqrt(1 / input_size)
        self.biases = np.random.randn(output_size, 1)
        
        # Momentum Terms
        self.vdW = np.zeros_like(self.weights)
        self.vdb = np.zeros_like(self.biases)

        # RMS Prop Terms
        self.sdW = np.zeros_like(self.weights)
        self.sdb = np.zeros_like(self.biases)


    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, input) + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate, _reg = False, _reg_lam = 0.2):
        
        M  = output_gradient.shape[1]

        input_gradient = np.dot(self.weights.T, output_gradient) 

        weight_update = np.dot(output_gradient, self.input.T) / M   # Why is the .T necessary
        bias_update   = np.sum(output_gradient, axis = 1, keepdims = True) / M

        if _reg == False:
            self.weights += - learning_rate * weight_update
        else:
            self.weights += - learning_rate * (weight_update + _reg_lam / M *self.weights)
        self.biases += - learning_rate * bias_update

        return input_gradient
    

    def backward_with_momentum(self, output_gradient, learning_rate, _reg = False, _reg_lam = 0.2, beta = 0.9):
        
        M  = output_gradient.shape[1]

        input_gradient = np.dot(self.weights.T, output_gradient) 

        
        bias_update_grad   = np.sum(output_gradient, axis = 1, keepdims = True) / M
        self.vdb = beta * self.vdb + (1 - beta)*bias_update_grad

        if _reg == False:
            weight_update_grad = np.dot(output_gradient, self.input.T) / M   
            self.vdW = beta * self.vdW + (1 - beta)*weight_update_grad
        else:
            weight_update_grad = np.dot(output_gradient, self.input.T) / M  + _reg_lam / M *self.weights
            self.vdW = beta * self.vdW + (1 - beta)*weight_update_grad
        
        self.biases += - learning_rate * self.vdb
        self.weights += - learning_rate * self.vdW
        
        return input_gradient


    def backward_with_Adam(self, output_gradient, learning_rate, iteration, _reg = False, _reg_lam = 0.2, beta_1 = 0.9, beta_2 = 0.999):
        
        M  = output_gradient.shape[1]

        input_gradient = np.dot(self.weights.T, output_gradient) 

        
        bias_update_grad   = np.sum(output_gradient, axis = 1, keepdims = True) / M
        self.vdb = beta_1 * self.vdb + (1 - beta_1)*bias_update_grad
        self.sdb = beta_2 * self.sdb + (1 - beta_2)*np.power(bias_update_grad, 2)

        # Bias Correction
        vdb_corr  = np.divide(self.vdb , 1 - np.power(beta_1, (iteration)))
        sdb_corr  = np.divide(self.sdb , 1 - np.power(beta_2, (iteration)))

        if _reg == False:
            weight_update_grad = np.dot(output_gradient, self.input.T) / M   
        else:
            weight_update_grad = np.dot(output_gradient, self.input.T) / M  + _reg_lam / M *self.weights
        
        self.vdW = beta_1 * self.vdW + (1 - beta_1) * weight_update_grad
        self.sdW = beta_2 * self.sdW + (1 - beta_2) * np.power(weight_update_grad, 2)
        
        # Bias Correction
        vdW_corr = np.divide(self.vdW , 1 - np.power(beta_1, (iteration)))
        sdW_corr = np.divide(self.sdW , 1 - np.power(beta_2, (iteration)))

        # Update
        self.biases += - learning_rate * np.divide(vdb_corr , np.sqrt(sdb_corr) + 1e-8)
        self.weights += - learning_rate * np.divide(vdW_corr , np.sqrt(sdW_corr) + 1e-8)
        
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
