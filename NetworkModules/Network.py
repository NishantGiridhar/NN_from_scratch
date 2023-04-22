# %%
import numpy as np
from NetworkModules import Activations
from NetworkModules.Layers import *



class Network:
    def __init__(self):
        self.layers = []
        self.loss   = None
        self.loss_derivative = None
        self.accuracy_function = None

    def add_accurace_function(self, func):
        self.accuracy_function = func
    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_function, loss_derivative):
        self.loss = loss_function
        self.loss_derivative = loss_derivative

    def load_data(self, x_train = None, y_train = None, x_test = None, y_test = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val  = x_test
        self.y_val  = y_test


    def _reinitalize_weights(self):
        for l, layer in enumerate(self.layers):
            if isinstance(layer, FullyConnectedLinearLayer):
                if self.layers[l+1].activation == Activations.relU:
                    layer.weights = np.random.randn(layer.weights.shape[0], layer.weights.shape[1]) * np.sqrt(2/layer.weights.shape[0] )
                elif self.layers[l+1].activation == Activations.tanh:
                    layer.weights = np.random.randn(layer.weights.shape[0], layer.weights.shape[1]) * np.sqrt(2/(layer.weights.shape[0] + layer.weights.shape[1]))


    def inference(self, x_pred):
        next_layer_input = x_pred
        for layer in self.layers:
            next_layer_input = layer.forward(next_layer_input)

        return next_layer_input

    def _batch_gradient_descent(self, learning_rate, epochs):
        
        for i in range(epochs):
            
            # Forward Pass
            next_layer_input = self.x_train
            for layer in self.layers:
                next_layer_input = layer.forward(next_layer_input)

            loss_value = self.loss(next_layer_input, self.y_train)

            # Backward Pass
            gradient_next = self.loss_derivative(next_layer_input, self.y_train)
            for layer in reversed(self.layers):
                gradient_next = layer.backward(gradient_next, learning_rate)

            if i%100 == 0:
                print(f'epoch {i+1}      error {loss_value}')
        return None
    

    def _stochastic_gradient_descent(self, learning_rate, epochs):
        samples = self.x_train.shape[1]

        self._iteration_summary(0, None)
        for i in range(epochs):
            error = 0
            for j in range(samples):
                next_layer_input = self.x_train[:,[j]]
                for layer in self.layers:
                    next_layer_input = layer.forward(next_layer_input)
                
                error += self.loss(next_layer_input, self.y_train[:,[j]]) 

                gradient_next = self.loss_derivative(next_layer_input, self.y_train[:,[j]])
                for layer in reversed(self.layers):
                    gradient_next = layer.backward(gradient_next, learning_rate)


            if (i%5 == 0) or (i == epochs-1):
                self._iteration_summary(i+1, error/samples)

        print('Training Complete')
        return None
                


    def _iteration_summary(self, iter, _error):
        if iter == 0:
            print('{:<10s}{:>5s}{:>20s}{:>20s}'.format('iter','Loss','Train Accuracy', 'Val Accuracy'))
        else:
            if self.x_val.any() != None:
                train_accuracy = round(self.accuracy_function(self.x_train, self.y_train), 2)
                val_accuracy = round(self.accuracy_function(self.x_val, self.y_val), 2)
            else:
                val_accuracy = 0
                train_accuracy = 0
            print('{:<10}{:>5}{:>20}{:>20}'.format(iter, round(_error,2), train_accuracy, val_accuracy))
# %%
