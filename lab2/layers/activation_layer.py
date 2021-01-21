from layers.layer import BaseLayer
import functions.activation_functions


class ActivationLayer(BaseLayer):
    def __init__(self, activation):
        self.weights=[]
        self.activation = activation
        self.activation_prime = getattr(functions.activation_functions, activation.__name__ + "_prime")

    def forward_propagation(self, input_data):
        """
        returns the activated input
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """
        Returns input_error=dE/dX for a given output_error=dE/dY.
        learning_rate is not used because there is no "learnable" parameters.
        """
        return self.activation_prime(self.input) * output_error
