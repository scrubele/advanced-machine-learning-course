import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


class Loss:

    def calculate(self):
        raise NotImplementedError

    def calculate_prime(self):
        raise NotImplementedError


class MeanSquareError(Loss):

    @staticmethod
    def calculate(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def calculate_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
