import numpy as np
from layer import layer

"""provides a linear layer for a neural network"""

class linlayer(layer):
    def __init__(self, activationFunction: str, inputSize: int, outputSize: int):
        """provides a linear layer for a neural network"""

        super().__init__(activationFunction=activationFunction)

        self.weights = layer.initializeWeights(outputSize, inputSize)
        self.bias = np.random.normal(loc=0, scale=1, size=(outputSize, ))

        self.inputSize = inputSize
    
    def forward(self, inputVec: np.ndarray) -> np.ndarray:
        """performs a forward pass"""

        # check for correct type
        if not isinstance(inputVec, np.ndarray):
            inputVec = np.array(inputVec)

        # check if shapes match
        if inputVec.shape[0] != self.inputSize:
            print("wrong input shape, shape must be ", self.inputSize)
            raise ValueError
        
        # multiply weights
        result = np.matmul(self.weights.transpose(), inputVec) 

        # add bias
        result += self.bias              

        # apply activation function and return result                   
        return self.activation(result)       

    def __call__(self, inputVec: np.ndarray):
        return self.forward(inputVec=inputVec)                   


if __name__ == '__main__':
    l = linlayer(activationFunction="relu", inputSize=2, outputSize=2)
    print(l(np.array([1, 2])))