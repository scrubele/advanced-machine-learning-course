from networks import DigitNetwork
from models import ThreeLayerlModel

if __name__ == "__main__":
    nn_trainer = DigitNetwork(
        learning_rate=0.02,
        max_epoch=10,
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        batch_size=128,
        batch_normalization=False,
        model=ThreeLayerlModel,
    )
    nn_trainer.compile()
    nn_trainer.fit()
    nn_trainer.evaluate()
    nn_trainer.validate()
