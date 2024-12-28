import numpy as np


def normalize(arr: np.ndarray) -> np.ndarray:
    """normalizes an array to values between 0 and 1"""

    # calculate min and max value of the array
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    # check for division by zero
    if arr_max == arr_min:
        return np.zeros(arr.shape)
    
    # normalize the array
    return (arr - arr_min) / (arr_max - arr_min)