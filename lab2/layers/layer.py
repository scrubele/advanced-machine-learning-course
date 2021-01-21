
class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        """
        computes the output Y of a layer for a given input X
        """
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        # computes dE/dX for a given dE/dY (and update parameters if any)
        raise NotImplementedError
