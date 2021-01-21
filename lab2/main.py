import numpy as np

from network import Network
from layers.dense_layer import DenseLayer
from layers.activation_layer import ActivationLayer
from functions.activation_functions import tanh, sigmoid
from functions.losses import MeanSquareError


class XORNetwork(Network):
    def __init__(self, epochs=1000, learning_rate=0.1):
        super(XORNetwork, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]], dtype=np.float64)
        self.y_train = np.array([[[0]], [[1]], [[1]], [[0]]], dtype=np.float64)
        self.initialize_parameters()
        self.create_network()

    def initialize_parameters(self):
        self.use(MeanSquareError)

    def create_network(self):
        input_nodes = 2
        output_nodes = 1
        self.add(DenseLayer(input_nodes, 3))
        self.add(ActivationLayer(sigmoid))
        self.add(DenseLayer(3, output_nodes))
        self.add(ActivationLayer(sigmoid))

    def predict(self, input_data):
        result = super(XORNetwork, self).predict(input_data)
        for i in range(0, len(result)):
            result[i] = result[i].tolist()[0][0]
        print("result", result)
        prediction = [1 if i>0.5 else 0 for i in result]
        print("prediction", prediction)

    def train(self):
        super(XORNetwork, self).fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )

    def test(self):
        result = self.predict(self.x_train)
        # print("result", result)


if __name__ == "__main__":
    xor_network = XORNetwork(epochs=5000, learning_rate=0.1)
    xor_network.train()
    xor_network.test()