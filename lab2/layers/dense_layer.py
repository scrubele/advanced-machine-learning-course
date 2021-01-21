from layers.layer import BaseLayer
import numpy as np

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size, learning_weights=[-0.5, 0.5]):
        """
        # input_size = number of input neurons
        # output_size = number of output neurons
        """
        self.weights = learning_weights[0]+ np.random.uniform(size=(input_size, output_size))*learning_weights[1]
        self.bias = np.random.uniform(size=(1, output_size))

    def forward_propagation(self, input_data):
        """
        returns output for a given input
        """
        self.input = input_data
        # print("input  ", '\t', self.input)
        # print("weights", '\t', self.weights)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """
        computes dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX.
        """
        input_error = np.dot(output_error, self.weights.T)
        # print(self.input, output_error)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error
        # update parameters
        self.weights += learning_rate * weights_error
        self.bias += np.sum(learning_rate * output_error, axis=0)
        return input_error
