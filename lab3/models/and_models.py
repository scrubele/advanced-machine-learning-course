import tensorflow as tf
import numpy as np


class ThreeLayerAndModel(tf.keras.Model):
    def __init__(self):
        super(ThreeLayerAndModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=2, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)


class ThreeLayerAndModelLinear(tf.keras.models.Sequential):
    def __init__(self):
        super(ThreeLayerAndModelLinear, self).__init__()
        self.add(
            tf.keras.layers.Dense(
                64, input_dim=2, activation=tf.keras.activations.linear
            )
        )
        self.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))
        self.dropout = tf.keras.layers.Dropout(0.5)
