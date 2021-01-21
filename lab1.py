import pandas as pd
import numpy as np
import itertools
from functools import reduce

DEBUG = False


def OR(*args):
    return any(args)


def XOR(*args):
    return reduce(lambda A, B: int(A) ^ int(B), args)


def AND(*args):
    return all(args)


class Perceptron:
    def __init__(
        self,
        number_of_inputs,
        weights_limits,
        train_nparray=None,
        learning_rate=0.01,
        max_epoch=300,
        with_bias=True,
    ):
        self.number_of_inputs = number_of_inputs
        self.weights_limits = weights_limits
        self.learning_rate = learning_rate
        self.train_nparray = train_nparray
        self.max_epoch = max_epoch
        self.with_bias = with_bias
        self.initialize_x_inputs()
        self.train_size = np.shape(self.x_inputs)[0]
        self.initialize_c_outputs()
        self.initialize_weights()
        self.initialize_bias()

    def initialize_weights(self):
        self.weights = np.random.randint(
            *self.weights_limits, size=self.number_of_inputs + 1
        )

    def initialize_bias(self):
        self.bias = np.dot(self.weights[0], self.x_inputs[0][0]) if self.with_bias else 0

    def initialize_x_inputs(self):
        self.x_inputs = self.train_nparray[:, :-1]

    def initialize_c_outputs(self):
        self.c_outputs = self.train_nparray[:, -1]
        print(f"Expected output: {self.c_outputs}")

    def get_convergence(self, expected_output_array, real_output_array):
        convergence = (expected_output_array == real_output_array).all()
        return convergence

    def activation_function(self, value):
        """
        The unit epoch activation funcion.
        """
        return 1 if value >= 0 else 0

    def predict(self, item):
        """
        Calculatatiion of the perceptron.
        """
        model = np.dot(self.x_inputs[item], self.weights) + self.bias
        logic = self.activation_function(model)
        return np.round(logic)

    def calculate_output(self):
        output = np.array([self.predict(item) for item in range(0, self.train_size)])
        return output

    def update_weights(self, real_output):
        for i in range(0, len(self.weights) + 1):
            error = self.c_outputs[i] - real_output[i]
            self.weights = np.add(
                self.weights,
                np.dot(
                    self.learning_rate * error,
                    self.x_inputs[i],
                ),
            )
        return self.weights

    def train(self):
        epoch = 0
        convergence = False
        self.initialize_bias()
        print(f"Initial weights: {self.weights} \t Bias: {self.bias}")
        while not convergence and epoch <= self.max_epoch:
            real_output = self.calculate_output()
            convergence = self.get_convergence(real_output, self.c_outputs)
            self.weights = self.update_weights(real_output)
            if DEBUG or convergence:
                print(
                    f"Epoch: {epoch} \t Weights: \t{self.weights} Output: \t {real_output}"
                )
            epoch += 1
            self.epoch = epoch


class LogicGatePerceptron(Perceptron):
    def __init__(
        self,
        number_of_inputs,
        logic_gate,
        weights_limits,
        learning_rate=0.02,
        max_epoch=300,
        with_bias=True,
    ):
        self.number_of_inputs = number_of_inputs
        self.logic_gate = logic_gate
        self.form_S_training_base_array()
        super(LogicGatePerceptron, self).__init__(
            number_of_inputs,
            weights_limits,
            self.train_nparray,
            learning_rate,
            max_epoch,
            with_bias,
        )

    def get_logic_operator_value(self, logic_gate, *args) -> bool:
        return globals()[logic_gate](*args)

    def form_S_training_base_array(self, max_value=1):
        x0 = 1
        training_base_list = []
        for x_input_values in itertools.product(
            range(max_value + 1), repeat=self.number_of_inputs
        ):
            x_inputs = (x0, *x_input_values)
            c_outputs = self.get_logic_operator_value(self.logic_gate, *x_input_values)
            training_base_list.append((*x_inputs, int(c_outputs)))
        self.train_nparray = np.array([*training_base_list])
        self.print_train_dataframe()

    def print_train_dataframe(self):
        train_dataframe = pd.DataFrame(
            data=self.train_nparray,
            columns=["x_" + str(i) for i in range(0, self.number_of_inputs + 1)]
            + ["c_" + self.logic_gate],
        )
        print("Train dataframe:\n", train_dataframe)


if __name__ == "__main__":
    perceptron = LogicGatePerceptron(
        number_of_inputs=2,
        logic_gate="OR",
        weights_limits=[-1, 1],
        learning_rate=0.02,
        max_epoch=100,
        with_bias=False,
   ).train()
