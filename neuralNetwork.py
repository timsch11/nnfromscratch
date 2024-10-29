from layer import layer
from linlayer import linlayer

import numpy as np

from lossFunctions import l2, dl2

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

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # configure layers for training
        for layer in self.layers:
            layer.activateTraining()

        result = self.predict(X=X)

        gradInput = self.lossFunctionDerivative(predicted=result, actual=y)

        for i in range(len(self.layers)-1, -1, -1):
            gradInput = self.layers[i].backpropagate(gradInput)

        # deactivate training configuration
        for layer in self.layers:
            layer.deactivateTraining()

    def predict(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)

        return X


if __name__ == '__main__':
    # nn = NN("l2", linlayer("relu", 4, 2, weights=np.array([[1., 4.], [2., 5.], [-1., 1.], [-2., 0]]), learningRate=0.005), linlayer("relu", 2, 1, weights=np.array([[1.], [1.]]), learningRate=0.005))
    nn = NN("l2", linlayer("relu", 4, 2, learningRate=0.001), linlayer("relu", 2, 1, learningRate=0.001))
    for i in range(50):
        print(nn.predict(X=np.array([1, 2, 0, 0])))
        nn.train(X=np.array([1, 2, 0, 0]), y=10)