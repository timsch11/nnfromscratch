import numpy as np
from layer import layer

"""provides a linear layer for a neural network"""

class linlayer(layer):
    def __init__(self, activationFunction: str, inputSize: int, outputSize: int, learningRate: float = 0.01):
        """provides a linear layer for a neural network"""

        super().__init__(activationFunction=activationFunction, learningRate=learningRate)

        self.weights = layer.initializeWeights(outputSize, inputSize)
        self.bias = np.random.normal(loc=0, scale=1, size=(outputSize, ))

        self.inputSize = inputSize

        self.training = False

        # data used for training later on
        self.inputVec = np.array([])
        self.resultBeforeActivation = np.array([])

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

        # store values for training
        if self.training:
            self.inputVec = inputVec
            self.resultBeforeActivation = result

        # apply activation function and return result                   
        return self.activation(result)  

    def activateTraining(self):
        self.training = True     
    
    def deactivateTraining(self):
        self.training = False     

    def backpropagate(self, dloss_dactFunc: np.ndarray) -> np.ndarray:
        """adjusts the weigths and biases of the current layer, returns input for backpropagtion of the next layer"""

        # derivative of loss function with respect to this layers results before activation
        dloss_dresult = self.derivativeActivation(xvec=self.resultBeforeActivation, previousDerivatives=dloss_dactFunc)

        # update weights#
        # TODO

        # update bias
        # TODO

        # find dloss_ input and return it
        # TODO

    def __call__(self, inputVec: np.ndarray):
        return self.forward(inputVec=inputVec)                   


if __name__ == '__main__':
    l = linlayer(activationFunction="relu", inputSize=2, outputSize=2)
    print(l(np.array([1, 2])))