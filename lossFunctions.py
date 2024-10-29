import numpy as np

from activationFunctions import applyMap

def l2(predicted: np.ndarray | int | list | int | float, actual: np.ndarray | int | list | int | float):
    # create new tensor with elemnt wise differnece between predicted and actual
    dif = predicted - actual
    
    # square each difference (element)
    return applyMap(xvec=dif, func=lambda x: x**2)

def dl2(predicted: np.ndarray | int | list | int | float, actual: np.ndarray| int | list | int | float):
    # return element wise dloss_dpredicted
    return 2*(predicted - actual)


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4]) 
    a += np.array([0, 1, 1, 4])
    print(a)