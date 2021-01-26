import tensorflow as tf
import numpy as np
from keras.layers.core import Dense


class ThreeLayerOrModelLinear(tf.keras.models.Sequential):
    def __init__(self):
        super(ThreeLayerOrModelLinear, self).__init__()
        self.add(Dense(64, input_dim=2, activation=tf.keras.activations.linear))
        self.add(Dense(1, activation=tf.keras.activations.linear))
        self.dropout = tf.keras.layers.Dropout(0.5)


class ThreeLayerOrModelTahn(tf.keras.models.Sequential):
    def __init__(self):
        super(ThreeLayerOrModelTahn, self).__init__()
        self.add(Dense(64, input_dim=2, activation=tf.keras.activations.tanh))
        self.add(Dense(1, activation=tf.keras.activations.sigmoid))
        self.dropout = tf.keras.layers.Dropout(0.5)


class ThreeLayerOrModelRelu(tf.keras.models.Sequential):
    def __init__(self):
        super(ThreeLayerOrModelRelu, self).__init__()
        self.add(Dense(16, input_dim=2, activation=tf.keras.activations.relu))
        self.add(Dense(1, activation=tf.keras.activations.sigmoid))
        self.dropout = tf.keras.layers.Dropout(0.5)


class FourLayerOrModel(tf.keras.models.Sequential):
    def __init__(self):
        super(FourLayerOrModel, self).__init__()
        self.add(Dense(2, input_dim=2, activation=tf.keras.activations.tanh))
        self.add(Dense(2, activation=tf.keras.activations.tanh))
        self.add(Dense(1, activation=tf.keras.activations.sigmoid))
        self.dropout = tf.keras.layers.Dropout(0.5)


class BigFourLayerOrModel(tf.keras.models.Sequential):
    def __init__(self):
        super(BigFourLayerOrModel, self).__init__()
        self.add(Dense(32, input_dim=2, activation=tf.keras.activations.relu))
        self.add(Dense(32, activation=tf.keras.activations.relu))
        self.add(Dense(1, activation=tf.keras.activations.sigmoid))
        self.dropout = tf.keras.layers.Dropout(0.5)
