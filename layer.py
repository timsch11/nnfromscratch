"""base class for all layers"""


import numpy as np
from activationFunctions import relu, sigmoid, tanh, softmax


# maps the name of an activation function (string) to corresponding value (function)
# if no function is specified the network should still work
activationFunctionMap = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "": lambda x: x
}


class layer:
    def __init__(self, activationFunction: str=""):
        """base class for all layers"""

        # check if activation function is available, raise Error and print available functions if not
        if activationFunction not in activationFunctionMap.keys():
            print("Your chosen activation function is not available, pick one out of ", activationFunctionMap.keys())
            raise KeyError
        
        # store activation function reference for later use
        self.activation = activationFunctionMap[activationFunction]

    @staticmethod
    def initializeWeights(inputSize: int, outputSize: int) -> np.ndarray: 
        # pytoch-like implementation, requires transpose
        return np.random.normal(loc=0, scale=1, size=(outputSize, inputSize)) 