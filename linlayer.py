import numpy as np
from layer import layer

"""provides a linear layer for a neural network"""

class linlayer(layer):
    def __init__(self, activationFunction: str, inputSize: int, outputSize: int, learningRate: float = 0.01, weights: np.ndarray=None):
        """provides a linear layer for a neural network"""

        # to be removed
        np.random.seed(1234)

        super().__init__(activationFunction=activationFunction, learningRate=learningRate)

        if weights is None:
            self.weights = layer.kaimingHeInitialization(inputSize=inputSize, outputSize=outputSize)
            self.bias = np.random.normal(loc=0, scale=1, size=(outputSize, ))  # np.array([0 for i in range(inputSize)])

        else:
            self.weights = weights
            self.bias = np.array([0 for i in range(outputSize)])

        self.inputSize = inputSize

        self.shape = (inputSize, outputSize)

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
  

    def backpropagate(self, dloss_dactFunc: np.ndarray, learningRate: float=0) -> np.ndarray:
        """adjusts the weigths and biases of the current layer, returns input for backpropagtion of the next layer (grad with respect to the previous layers activation)"""

        # set learning rate to class standard if not specified
        if learningRate == 0:
            learningRate = self.learningRate

        # derivative of loss function with respect to this layers results before activation
        dloss_dresult = self.derivativeActivation(xvec=self.resultBeforeActivation, previousDerivatives=dloss_dactFunc)

        # init empty vector for dloss_dpreviousactivation
        dloss_dinput = np.array([0 for i in range(self.shape[0])])

        # update weights and store dloss_dpreviousactivation
        for j in range(self.shape[1]):  # ...for every neuron

            # init empty list to keep track of the derivatives with respect to the input
            der_prevAct = []

            for i in range(self.shape[0]):  # ...for every weight

                # derivative of loss function with respect to the current weight
                dloss_dweight = dloss_dresult[j] * self.inputVec[i]

                # update dloss_dpreviousactivation
                der_prevAct.append(dloss_dresult[j] * self.weights[i][j])

                # update weights
                self.weights[i][j] = self.weights[i][j] - (learningRate * dloss_dweight)

            # update dloss_dpreviousactivation
            dloss_dinput = dloss_dinput + np.array(der_prevAct)

        # update bias
        self.bias = self.bias - (learningRate * dloss_dactFunc)

        # return dloss_dinput (grad with respect to the previous layers activations)
        return dloss_dinput
    

    def __call__(self, inputVec: np.ndarray) -> np.ndarray:
        """calls func forward"""

        return self.forward(inputVec=inputVec)                   


if __name__ == '__main__':
    weights = np.array([1, 2])
    a = weights.transpose()
    print(weights)
    exit()
    l = linlayer(activationFunction="relu", inputSize=2, outputSize=2)
    print(l(np.array([1, 2])))