import numpy as np
import math

from typing import Callable

"""implements all supported activation functions, all of which are designed to either be fed with a single number or a vector of numbers (numpy.array) and return the result as a numpy array"""


def applyMap(xvec: np.ndarray | list | int | float, func: Callable) -> np.ndarray:
    """applies the specified R -> R function to all elements of the specified vector (R^n -> R^n)"""

    # ck if xvec is a single number and convert to numpy array if necessary
    if type(xvec) in {int, float}:  # if xvec is a single number
        xvec = np.array([xvec])     # convert it to a numpy array

    # create a map that applies f to all elements
    map = np.vectorize(func)            

    # return result of that map when applied to input
    return map(xvec)                    

def relu(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies relu function to all elements in a numpy array, returns numpy array of same shape"""

    # define relu function as single variable function
    f = lambda x: x if x > 0 else 0

    # apply Map to all elements of the vector
    return applyMap(xvec=xvec, func=f)


def drelu(xvec: np.ndarray | list | int | float, postActivation: np.ndarray, previousDerivatives: np.ndarray=None) -> np.ndarray:
    """derivative of relu function with respect to input, returns vector of dloss_dreluinput, xvec := input of relu function"""

    # check if xvec is a single number and convert to numpy array if necessary
    if type(xvec) in {int, float}:  # if xvec is a single number
        xvec = np.array([xvec])     # convert it to a numpy array
    
    # define relu derivative as single variable function
    df_dx = lambda x: 1 if x > 0 else 0

    # if previousDerivatives is empty or do not exist it should have no impact on result
    if previousDerivatives is None:
        return applyMap(xvec=xvec, func=df_dx)

    # apply Map to all elements of the vector
    return np.multiply(applyMap(xvec=xvec, func=df_dx), previousDerivatives)


def sigmoid(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies sigmoid function to all elements in a numpy array, returns numpy array of same shape"""
    
    # define tanh function as single variable function
    f = lambda x: 1 / (1 + math.exp(-x))
    
    # apply Map to all elements of the vector
    return applyMap(xvec=xvec, func=f)


def dsigmoid(xvec: np.ndarray | list | int | float, postActivation: np.ndarray, previousDerivatives: np.ndarray=None) -> np.ndarray:
    """derivative of sigmoid function with respect to input, returns vector of dloss_dsigmoidinput, xvec := input of sigmoid function"""

    # check if xvec is a single number and convert to numpy array if necessary
    if type(xvec) in {int, float}:  # if xvec is a single number
        xvec = np.array([xvec])     # convert it to a numpy array
    
    # define sigmoid derivative as single variable function
    df_dx = lambda x: x * (1 - x)

    # if previousDerivatives is empty or do not exist it should have no impact on result
    if previousDerivatives is None:
        return applyMap(xvec=postActivation, func=df_dx)

    # apply Map to all elements of the vector
    return np.multiply(applyMap(xvec=postActivation, func=df_dx), previousDerivatives)


def tanh(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies tanh function to all elements in a numpy array, returns numpy array of same shape"""
    
    # define tanh function as single variable function
    f = lambda x: (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    
    # apply Map to all elements of the vector
    return applyMap(xvec=xvec, func=f)


def dtanh(xvec: np.ndarray | list | int | float, postActivation: np.ndarray, previousDerivatives: np.ndarray=None) -> np.ndarray:
    """derivative of tanh function with respect to input, returns vector of dloss_dtanhinput, xvec := input of tanh function"""

    # check if xvec is a single number and convert to numpy array if necessary
    if type(xvec) in {int, float}:  # if xvec is a single number
        xvec = np.array([xvec])     # convert it to a numpy array
    
    # define sigmoid derivative as single variable function
    df_dx = lambda x: 1 - x**2

    # if previousDerivatives is empty or do not exist it should have no impact on result
    if previousDerivatives is None:
        return applyMap(xvec=postActivation, func=df_dx)

    # apply Map to all elements of the vector
    return np.multiply(applyMap(xvec=postActivation, func=df_dx), previousDerivatives)


def softmax(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies softmax function to all elements in a numpy array, returns numpy array of same shape"""
    
    # calculate sum of divisor
    s = 0                           
    for el in xvec:
        s += math.exp(el)

    # define softmax function as single variable function
    f = lambda x: math.exp(x) / s

    # apply Map to all elements of the vector
    return applyMap(xvec=xvec, func=f)


def dsoftmax(xvec: np.ndarray | list | int | float, postActivation: np.ndarray, previousDerivatives: np.ndarray=None):
    """derivative of softmax function with respect to input, returns vector of dloss_dsoftmaxinput, xvec := input of softmax function"""

    # check if xvec is a single number and convert to numpy array if necessary
    if type(xvec) in {int, float}:  # if xvec is a single number
        xvec = np.array([xvec])     # convert it to a numpy array

    # define softmax derivative as single variable function
    def df_dx(x, i, j):
        if i == j:
            return x[i] * (1 - x[i])
        else:
            return -x[i] * x[j]

    # if previousDerivatives is empty or do not exist it should have no impact on result
    if previousDerivatives is None:
        previousDerivatives = np.ones_like(postActivation)

    # calculate the Jacobian matrix of the softmax function
    jacobian_matrix = np.zeros((len(postActivation), len(postActivation)))
    for i in range(len(postActivation)):
        for j in range(len(postActivation)):
            jacobian_matrix[i][j] = df_dx(postActivation, i, j)

    # multiply the Jacobian matrix by the previous derivatives
    return np.dot(jacobian_matrix, previousDerivatives)
