import numpy as np

from network import Network
from layers.dense_layer import DenseLayer
from layers.activation_layer import ActivationLayer
from functions.activation_functions import tanh, sigmoid, relu
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
        self.add(DenseLayer(input_nodes, 2))
        self.add(ActivationLayer(relu))
        self.add(DenseLayer(2, 5))
        self.add(ActivationLayer(relu))
        self.add(DenseLayer(5, 3))
        self.add(ActivationLayer(relu))
        # self.add(DenseLayer(3, 2))
        # self.add(ActivationLayer(sigmoid))
        self.add(DenseLayer(3, output_nodes))
        self.add(ActivationLayer(relu))

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
        return result


if __name__ == "__main__":
    xor_network = XORNetwork(epochs=500, learning_rate=0.02)
    wished_result = [0.0,1.0,1.0,0.0]
    isResult = False
    time = 0
    while(not isResult):
        print("Time "+str(time))
        xor_network.train()
        result = xor_network.test()
        if(isResult==wished_result):
            isResult=True
        print("----")
        time+=1
        if time>2:
            isResult = True

