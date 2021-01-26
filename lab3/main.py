from operator import or_, and_, xor, not_

from networks import LogicGateNetwork
from models.and_models import ThreeLayerAndModel, ThreeLayerAndModelLinear
from models.or_models import (
    ThreeLayerOrModelLinear,
    ThreeLayerOrModelRelu,
    ThreeLayerOrModelTahn,
    FourLayerOrModel,
    BigFourLayerOrModel,
)
from models.xor_models import ThreeLayerXorModel, FourLayerXorModel

if __name__ == "__main__":
    nn_trainer = LogicGateNetwork(
        logic_gate=xor,
        learning_rate=0.02,
        max_epoch=600,
        model=FourLayerXorModel,
    )
    nn_trainer.compile()
    nn_trainer.fit()
    nn_trainer.validate()
