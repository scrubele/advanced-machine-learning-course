import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    The sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
    The first derivative of the sigmoid function x
    """
    return x * (1 - x)


def tanh(x):
    """
    The tanh activation function.
    """
    return np.tanh(x)


def tanh_prime(x):
    """
    The first derivative of the tanh function x
    """
    # print(x, x**2)
    return 1.0 - x ** 2
