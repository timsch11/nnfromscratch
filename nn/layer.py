import numpy as np
from nn.activationFunctions import relu, sigmoid, tanh, softmax, drelu


# maps the name of an activation function (string) to corresponding value (function)
# if no function is specified the network should still work
activationFunctionMap = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "": lambda x: x
}

# maps the name of the activation function to its derivative
derivativeActivationFunctionMap = {
    "relu": drelu,
    "": lambda x: 1
}


"""base class for all layers"""
class layer:
    def __init__(self, activationFunction: str="", learningRate: float = 0.01):
        """base class for all layers"""

        # check if activation function is available, raise Error and print available functions if not
        if activationFunction not in activationFunctionMap.keys():
            print("Your chosen activation function is not available, pick one out of ", activationFunctionMap.keys())
            raise KeyError
        
        # store activation function (and its derivative) reference for later use
        self.activation = activationFunctionMap[activationFunction]
        self.derivativeActivation = derivativeActivationFunctionMap[activationFunction]

        # are we performing a training iteration?
        self.training = False

        # store learning rate for training
        self.learningRate = learningRate

    def activateTraining(self):
        self.training = True     
    
    def deactivateTraining(self):
        self.training = False   

    @staticmethod
    def kaimingHeInitialization(inputSize: int, outputSize: int) -> np.ndarray:
        return np.random.normal(loc=0, scale=(2/inputSize)**(1/2), size=(inputSize, outputSize)) 

    @staticmethod
    def xavierInitialization(inputSize: int, outputSize: int) -> np.ndarray:
        return np.random.normal(loc=0, scale=(1/inputSize)**(1/2), size=(inputSize, outputSize)) 

    @staticmethod
    def initializeWeights(inputSize: int, outputSize: int) -> np.ndarray: 
        # pytoch-like implementation, requires transpose
        return np.random.normal(loc=0, scale=1, size=(inputSize, outputSize)) 
    