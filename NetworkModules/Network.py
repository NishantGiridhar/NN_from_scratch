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

    def add_layer(self, layer, dropout_layer = False, keep_prob = 1):
        self.layers.append(layer)
        if dropout_layer == True and isinstance(layer, FullyConnectedLinearLayer):
            raise NotImplementedError
        else:
            if dropout_layer == True:
                self.layers[-1]._add_dropout(keep_prob)


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
    

    def _stochastic_gradient_descent(self, learning_rate, epochs, L2_reg = False, L2_lambda = 0.2):
        samples = self.x_train.shape[1]

        self._iteration_summary(0, None)
        for i in range(epochs):
            error = 0
            for j in range(samples):
                # Forward Pass
                next_layer_input = self.x_train[:,[j]]
                for layer in self.layers:
                    next_layer_input = layer.forward(next_layer_input)
                    
                    if isinstance(layer, ActivationLayer):
                        dropout_mask = np.random.rand(next_layer_input.shape[0], next_layer_input.shape[1]) < layer.keep_prob
                        next_layer_input = np.multiply(next_layer_input, dropout_mask) / layer.keep_prob
                

                if L2_reg == False:
                    L2_loss = 0
                else:
                    L2_loss = L2_lambda / 2 * np.sum(np.linalg.norm(layer.weights, 'fro') for layer in self.layers if isinstance(layer, FullyConnectedLinearLayer)) 
                error += self.loss(next_layer_input, self.y_train[:,[j]]) + L2_loss

                # Backward Pass
                gradient_next = self.loss_derivative(next_layer_input, self.y_train[:,[j]])
                for layer in reversed(self.layers):
                    gradient_next = layer.backward(gradient_next, learning_rate, L2_reg, L2_lambda)


            if (i%5 == 0) or (i == epochs-1):
                self._iteration_summary(i+1, error/samples)

        print('Training Complete')
        return None
    


    def _mini_batch_gradient_descent(self, batch_size, learning_rate, epochs, L2_reg = False, L2_lambda = 0.2, rate_decay = 1, seed = None):
        batches = self._shuffle_mini_batches(self.x_train, self.y_train, batch_size, seed)
        N_batches = len(batches)

        self._iteration_summary(0, None)

        for i in range(epochs):
            error = 0
            decayed_learning_rate = rate_decay**i * learning_rate
            for j in range(N_batches):
                batch_size = batches[j][0].shape[1]

                # Forward Pass
                next_layer_input = batches[j][0]
                for layer in self.layers:
                
                    next_layer_input = layer.forward(next_layer_input)
                
                    if isinstance(layer, ActivationLayer):
                        dropout_mask = np.random.rand(next_layer_input.shape[0], next_layer_input.shape[1]) < layer.keep_prob
                        next_layer_input = np.multiply(next_layer_input, dropout_mask) / layer.keep_prob
                    
                if L2_reg == False:
                    L2_loss = 0
                else:
                    L2_loss = L2_lambda / 2 /batch_size * np.sum(np.linalg.norm(layer.weights, 'fro') for layer in self.layers if isinstance(layer, FullyConnectedLinearLayer)) 
                
                error += self.loss(next_layer_input, batches[j][1]) + L2_loss
                
                # Backward Pass
                gradient_next = self.loss_derivative(next_layer_input, batches[j][1])
                for layer in reversed(self.layers):
                    gradient_next = layer.backward(gradient_next, decayed_learning_rate, L2_reg, L2_lambda)

            if (i%5 == 0) or (i == epochs-1):
                self._iteration_summary(i+1, error/N_batches)
        
        print('Training Complete')
        return None
    
    def _mini_batch_gradient_descent_with_momentum(self, batch_size, learning_rate, epochs, L2_reg = False, L2_lambda = 0.2, rate_decay = 1, seed = None, beta = 0.9):
        batches = self._shuffle_mini_batches(self.x_train, self.y_train, batch_size, seed)
        N_batches = len(batches)

        self._iteration_summary(0, None)

        for i in range(epochs):
            error = 0
            decayed_learning_rate = rate_decay**i * learning_rate
            for j in range(N_batches):
                batch_size = batches[j][0].shape[1]

                # Forward Pass
                next_layer_input = batches[j][0]
                for layer in self.layers:
                
                    next_layer_input = layer.forward(next_layer_input)
                
                    if isinstance(layer, ActivationLayer):
                        dropout_mask = np.random.rand(next_layer_input.shape[0], next_layer_input.shape[1]) < layer.keep_prob
                        next_layer_input = np.multiply(next_layer_input, dropout_mask) / layer.keep_prob
                    
                if L2_reg == False:
                    L2_loss = 0
                else:
                    L2_loss = L2_lambda / 2 /batch_size * np.sum(np.linalg.norm(layer.weights, 'fro') for layer in self.layers if isinstance(layer, FullyConnectedLinearLayer)) 
                
                error += self.loss(next_layer_input, batches[j][1]) + L2_loss
                
                # Backward Pass
                gradient_next = self.loss_derivative(next_layer_input, batches[j][1])
                for layer in reversed(self.layers):
                    if isinstance(layer, FullyConnectedLinearLayer):
                        gradient_next = layer.backward_with_momentum(gradient_next, decayed_learning_rate, L2_reg, L2_lambda, beta)
                    else:
                        gradient_next = layer.backward(gradient_next, decayed_learning_rate, L2_reg, L2_lambda)
            if (i%5 == 0) or (i == epochs-1):
                self._iteration_summary(i+1, error/N_batches)
        
        print('Training Complete')
        return None
    
    def _mini_batch_Adam(self, batch_size, learning_rate, epochs, L2_reg = False, L2_lambda = 0.2, rate_decay = 1, seed = None, beta_1 = 0.9, beta_2 = 0.999, adaptive_learning_rate = 'no', decay_rate=0.3, schedule_interval=10):
        batches = self._shuffle_mini_batches(self.x_train, self.y_train, batch_size, seed)
        N_batches = len(batches)

        self._iteration_summary(0, None, None)
        t = 0
        for i in range(epochs):
            error = 0

            if adaptive_learning_rate == 'exponential decay':
                decayed_learning_rate = rate_decay**i * learning_rate
            elif adaptive_learning_rate == 'scheduled':
                decayed_learning_rate = 1 / (1 + decay_rate * np.floor(i/schedule_interval)) * learning_rate
            elif adaptive_learning_rate == 'no':
                decayed_learning_rate = learning_rate
            else:
                print('No such learning rate strategy')
                raise NotImplementedError

            for j in range(N_batches):
                t += 1

                batch_size = batches[j][0].shape[1]


                #####################  Forward Propagation
                next_layer_input = batches[j][0]
                for layer in self.layers:
                
                    next_layer_input = layer.forward(next_layer_input)
                
                    if isinstance(layer, ActivationLayer):
                        dropout_mask = np.random.rand(next_layer_input.shape[0], next_layer_input.shape[1]) < layer.keep_prob
                        next_layer_input = np.multiply(next_layer_input, dropout_mask) / layer.keep_prob
                
                #####################  L2 regularization
                if L2_reg == False:
                    L2_loss = 0
                else:
                    L2_loss = L2_lambda / 2 /batch_size * np.sum(np.linalg.norm(layer.weights, 'fro') for layer in self.layers if isinstance(layer, FullyConnectedLinearLayer)) 
                
                error += self.loss(next_layer_input, batches[j][1]) + L2_loss
                
                #####################  Backward Propagation
                gradient_next = self.loss_derivative(next_layer_input, batches[j][1])
                for layer in reversed(self.layers):
                    if isinstance(layer, FullyConnectedLinearLayer):
                        gradient_next = layer.backward_with_Adam(gradient_next, decayed_learning_rate, int(t), L2_reg, L2_lambda, beta_1, beta_2)
                    else:
                        gradient_next = layer.backward(gradient_next, decayed_learning_rate, L2_reg, L2_lambda)
            if (i%5 == 0) or (i == epochs-1):
                self._iteration_summary(i+1, error/N_batches, decayed_learning_rate)
        
        print('Training Complete')
        return None





    def _shuffle_mini_batches(self, X, Y, batch_size=64, seed = None):
        np.random.seed(seed)

        M = X.shape[1]
        mini_batches = []

        # Shuffle X, Y
        permutation = list(np.random.permutation(M))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        inc = batch_size

        num_complete_minibatches = int(np.floor(M / inc))

        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:,k*inc:(k+1)*inc]
            mini_batch_Y = shuffled_Y[:,k*inc:(k+1)*inc]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        if M % inc != 0:
                mini_batch_X = shuffled_X[:,(num_complete_minibatches)*inc:]
                mini_batch_Y = shuffled_Y[:,(num_complete_minibatches)*inc:]
                
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)
        
        return mini_batches


    def _iteration_summary(self, iter, _error, learning_rate):
        if iter == 0:
            print('{:<10s}{:>5s}{:>20s}{:>20s}{:>20s}'.format('iter','Loss', 'learn rate', 'Train Accuracy', 'Val Accuracy'))
        else:
            if self.x_val.any() != None:
                train_accuracy = round(self.accuracy_function(self.x_train, self.y_train), 2)
                val_accuracy = round(self.accuracy_function(self.x_val, self.y_val), 2)
            else:
                val_accuracy = 0
                train_accuracy = 0
            print('{:<10}{:>5}{:>20}{:>20}{:>20}'.format(iter, round(_error,2), round(learning_rate, 5), train_accuracy, val_accuracy))
# %%
