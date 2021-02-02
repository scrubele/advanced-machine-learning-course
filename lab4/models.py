import tensorflow as tf
import numpy as np
from keras.layers.core import Dense


class ThreeLayerlModel(tf.keras.models.Sequential):
    def __init__(self, batch_normalization=False, dropout=True):
        super(ThreeLayerlModel, self).__init__()
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        if batch_normalization:
            self.add(tf.keras.layers.BatchNormalization())
        if dropout:
            self.add(tf.keras.layers.Dropout(0.5))
        self.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        if batch_normalization:
            self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
