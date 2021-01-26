import tensorflow as tf
import numpy as np
from keras.layers.core import Dense


class ThreeLayerXorModel(tf.keras.models.Sequential):
    def __init__(self):
        super(ThreeLayerXorModel, self).__init__()
        self.add(Dense(16, input_dim=2, activation=tf.keras.activations.relu))
        self.add(Dense(1, activation=tf.keras.activations.sigmoid))
        self.dropout = tf.keras.layers.Dropout(0.5)


class FourLayerXorModel(tf.keras.models.Sequential):
    def __init__(self):
        super(FourLayerXorModel, self).__init__()
        self.add(Dense(5, input_dim=2, activation=tf.keras.activations.relu))
        self.add(Dense(3, activation=tf.keras.activations.relu))
        self.add(Dense(1, activation=tf.keras.activations.relu))
        self.dropout = tf.keras.layers.Dropout(0.5)
