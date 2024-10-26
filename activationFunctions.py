import numpy as np
import math

from typing import Callable

"""implements all supported activation functions, all of which are designed to either be fed with a single number or a vector of numbers (numpy.array) and return the result as a numpy array"""

def applyMap(xvec: np.ndarray | list | int | float, func: Callable) -> np.ndarray:
    """applies the specified R -> R function to all elements of the specified vector (R^n -> R^n)"""

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


def sigmoid(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies sigmoid function to all elements in a numpy array, returns numpy array of same shape"""
    
    # define tanh function as single variable function
    f = lambda x: 1 / (1 + math.exp(-x))
    
    # apply Map to all elements of the vector
    return applyMap(xvec=xvec, func=f)


def tanh(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies tanh function to all elements in a numpy array, returns numpy array of same shape"""
    
    # define tanh function as single variable function
    f = lambda x: (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    
    # apply Map to all elements of the vector
    return applyMap(xvec=xvec, func=f)


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


if __name__ == '__main__':
    print(relu([1.2, 1, 5]))