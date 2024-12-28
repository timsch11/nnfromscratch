from nn.layer import layer
from nn.linlayer import linlayer

import numpy as np

from nn.lossFunctions import l2, dl2

from nn.util import normalize

"""class for managing a whole neural network"""


lossFunctionMap = {
    "l2": l2
}

# maps the name of the activation function to its derivative
lossFunctionDerivativeMap = {
    "l2": dl2
}


class NN:
    def __init__(self, lossFunction: str, *layers: layer) -> None:
        """creates a new neural network, a loss function and an arbitrary amount of layers can be configured"""

        # create references to loss function and corresponding derivatives
        self.lossFunction = lossFunctionMap[lossFunction]
        self.lossFunctionDerivative = lossFunctionDerivativeMap[lossFunction]

        # store layer in a list
        self.layers = layers

    def train(self, X: np.ndarray, y: np.ndarray, learningRate: float=0.01) -> None:
        """trains the neural network based on one given sample"""

        # configure layers for training
        for layer in self.layers:
            layer.activateTraining()

        # get prediction based on current weights
        result = self.predict(X=X)

        # dloss/dpredicted
        gradInput = self.lossFunctionDerivative(predicted=result, actual=y)

        # perform backpropagation
        for i in range(len(self.layers)-1, -1, -1):
            gradInput = self.layers[i].backpropagate(gradInput, learningRate)

        # deactivate training configuration
        for layer in self.layers:
            layer.deactivateTraining()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """performs the chained forward pass for all layers"""

        # normalize input 
        X = normalize(X)

        # performs forward pass for all layers
        for layer in self.layers:
            X = layer.forward(X)

        return X
    