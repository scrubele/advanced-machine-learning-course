import tensorflow as tf
import numpy as np
import datetime
from operator import or_, and_, xor, not_


class Network:
    def __init__(
        self,
        model,
        x_train,
        y_train,
        learning_rate,
        max_epoch=600,
        loss=tf.keras.losses.mean_squared_error,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.loss = loss
        self.model = model()
        self.init_tensorboard()

    def init_tensorboard(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

    def compile(self):
        self.model.compile(
            loss=self.loss,
            optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate),
            metrics=[tf.keras.metrics.binary_accuracy],
        )

    def fit(self):
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.max_epoch,
            verbose=2,
            callbacks=[self.tensorboard_callback],
        )

    def validate(self):
        # Validation
        print("Round off Values: \n", self.model.predict(self.x_train).round())
        print("Actual Values: \n", self.model.predict(self.x_train))


class LogicGateNetwork(Network):
    def __init__(
        self,
        model,
        learning_rate,
        logic_gate=or_,
        max_epoch=600,
        loss=tf.keras.losses.mean_squared_error,
    ):
        self.logic_gate = logic_gate
        self.x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.form_y_train()
        super(LogicGateNetwork, self).__init__(
            x_train=self.x_train,
            y_train=self.y_train,
            learning_rate=learning_rate,
            max_epoch=max_epoch,
            loss=loss,
            model=model,
        )

    def form_y_train(self):
        self.y_train = np.array([[self.logic_gate(x, y)] for (x, y) in self.x_train])
        print(self.y_train)
