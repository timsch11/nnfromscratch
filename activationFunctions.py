import numpy as np
import math


"""implements all supported activation functions, all of which are designed to either be fed with a single number or a vector of numbers (numpy.array) and return the result as a numpy array"""


def relu(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies relu function to all elements in a numpy array, returns numpy array of same shape"""
    
    f = lambda x: x if x > 0 else 0     # define mere element wise map 
    map = np.vectorize(f)               # create a map that applies f to all elements
    return map(xvec)                    # return result of that map when applied to input


def sigmoid(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies sigmoid function to all elements in a numpy array, returns numpy array of same shape"""
    
    f = lambda x: 1 / (1 + math.exp(-x))    # define mere element wise map 
    map = np.vectorize(f)                   # create a map that applies f to all elements
    return map(xvec)                        # return result of that map when applied to input


def tanh(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies tanh function to all elements in a numpy array, returns numpy array of same shape"""
    
    f = lambda x: (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))    # define mere element wise map 
    map = np.vectorize(f)                                                        # create a map that applies f to all elements
    return map(xvec)                                                             # return result of that map when applied to input


def softmax(xvec: np.ndarray | list | int | float) -> np.ndarray:
    """applies softmax function to all elements in a numpy array, returns numpy array of same shape"""
    
    # calculate sum of divisor
    s = 0                           
    for el in xvec:
        s += math.exp(el)

    f = lambda x: math.exp(x) / s    # define mere element wise map 
    map = np.vectorize(f)            # create a map that applies f to all elements
    return map(xvec)   


if __name__ == '__main__':
    print(type(softmax([1.2, 1, 5])))