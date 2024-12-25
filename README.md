# Neural Network from Scratch

This project implements a basic neural network framework in Python, built from scratch using only numpy arrays. The implementation includes fundamental components such as layers, activation functions, and loss functions.
This was a short personal project to enhace my understanding of neural networks.

## Features

- Custom neural network implementation with configurable layers
- Support for multiple activation functions:
    - ReLU
    - Sigmoid
    - Tanh
    - Softmax
- L2 loss function
- Flexible layer architecture
- Kaiming He and Xavier weight initialization

## Project Structure

- `neuralnetwork.py`: Main neural network class implementation
- `layer.py`: Base class for network layers
- `linlayer.py`: Linear layer implementation
- `activationFunctions.py`: Implementation of activation functions
- `lossFunctions.py`: Implementation of loss functions

## Usage Example

```python
from neuralnetwork import NN
from linlayer import linlayer
import numpy as np

# Create a neural network with two layers
nn = NN("l2", 
        linlayer("relu", 4, 2, learningRate=0.0001), 
        linlayer("relu", 2, 1, learningRate=0.0001)
)

# Train the network
X = np.array([1, 2, 0, 0])
y = 10
nn.train(X=X, y=y)

# Make predictions
prediction = nn.predict(X=np.array([1, 2, 0, 0]))
```

## Dependencies

- NumPy
- Python 3.10 or newer (works for older ones as well, however the type hinting (| -> Union(...)) must be changed)
