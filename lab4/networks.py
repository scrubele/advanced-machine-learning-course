import tensorflow as tf # Import tensorflow library
import matplotlib.pyplot as plt # Import matplotlib library
import numpy as np
import datetime



class Network:
    def __init__(
        self,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        learning_rate,
        max_epoch=600,
        loss=tf.keras.losses.mean_squared_error,
        optimizer="adam",
        batch_size=200,
        batch_normalization=False,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.model = model(dropout=False, batch_normalization=batch_normalization)
        self.init_tensorboard()

    def init_tensorboard(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

    def compile(self):
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )

    def fit(self):
        self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test,self.y_test),
            epochs=self.max_epoch,
            batch_size=self.batch_size,
            verbose=2,
            callbacks=[self.tensorboard_callback],
        )

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(x=self.x_test, y=self.y_test)
        print('\nTest accuracy:', test_acc)

    def validate(self):
        predictions = np.argmax(self.model.predict(self.x_test), axis=-1) # Make prediction
        print(np.argmax(predictions[1000])) # Print out the number
        plt.imshow(self.x_test[1000], cmap="gray") # Import the image
        plt.savefig("x_test.png")


class DigitNetwork(Network):
    def __init__(
        self,
        model,
        learning_rate,
        max_epoch=600,
        loss=tf.keras.losses.mean_squared_error,
        optimizer="adam",
        batch_size=200,
        batch_normalization=False,
    ):
        self.prepare_dataset()
        super(DigitNetwork, self).__init__(
            model=model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            learning_rate=learning_rate,
            max_epoch=max_epoch,
            loss=loss,
            optimizer=optimizer,
            batch_size=batch_size,
            batch_normalization=batch_normalization,
        )


    def load_dataset(self):
        mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
        (x_train, y_train),(x_test, y_test) = mnist.load_data() # Load data
        return (x_train, y_train),(x_test, y_test)


    def normalize_dataset(self, x_train, x_test):
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)
        return x_train, x_test

    def prepare_dataset(self):
        (self.x_train, self.y_train),(self.x_test, self.y_test) = self.load_dataset()
        self.x_train, self.x_test =self.normalize_dataset(self.x_train, self.x_test)
