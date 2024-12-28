import unittest
import numpy as np
from nn.linlayer import linlayer


class TestLinLayer(unittest.TestCase):
    def setUp(self):
        self.input_size = 3
        self.output_size = 2
        self.layer = linlayer(
            activationFunction="relu",
            inputSize=self.input_size,
            outputSize=self.output_size
        )

    def test_initialization(self):
        """Test layer initialization"""
        self.assertEqual(self.layer.inputSize, self.input_size)
        self.assertEqual(self.layer.shape, (self.input_size, self.output_size))
        self.assertEqual(self.layer.weights.shape, (self.input_size, self.output_size))
        self.assertEqual(self.layer.bias.shape, (self.output_size,))
        self.assertEqual(len(self.layer.inputVec), 0)
        self.assertEqual(len(self.layer.resultBeforeActivation), 0)

    def test_custom_weights_initialization(self):
        """Test initialization with custom weights"""
        custom_weights = np.ones((self.input_size, self.output_size))
        layer = linlayer(
            activationFunction="relu",
            inputSize=self.input_size,
            outputSize=self.output_size,
            weights=custom_weights
        )
        self.assertTrue(np.array_equal(layer.weights, custom_weights))
        self.assertTrue(np.array_equal(layer.bias, np.zeros(self.output_size)))

    def test_forward_basic(self):
        """Test basic forward pass"""
        input_vec = np.array([1., 2., 3.])
        output = self.layer.forward(input_vec)
        self.assertEqual(output.shape, (self.output_size,))

    def test_forward_list_input(self):
        """Test forward pass with list input"""
        input_vec = [1., 2., 3.]
        output = self.layer.forward(input_vec)
        self.assertEqual(output.shape, (self.output_size,))

    def test_forward_invalid_shape(self):
        """Test forward pass with invalid input shape"""
        input_vec = np.array([1., 2.])  # Wrong size
        with self.assertRaises(ValueError):
            self.layer.forward(input_vec)

    def test_forward_training_mode(self):
        """Test forward pass in training mode"""
        self.layer.activateTraining()
        input_vec = np.array([1., 2., 3.])
        output = self.layer.forward(input_vec)
        self.assertTrue(np.array_equal(self.layer.inputVec, input_vec))
        self.assertEqual(self.layer.resultBeforeActivation.shape, (self.output_size,))

    def test_backpropagation(self):
        """Test backpropagation"""
        self.layer.activateTraining()
        input_vec = np.array([1., 2., 3.])
        self.layer.forward(input_vec)
        
        dloss = np.array([0.1, 0.2])
        old_weights = self.layer.weights.copy()
        old_bias = self.layer.bias.copy()
        
        grad = self.layer.backpropagate(dloss)
        
        self.assertEqual(grad.shape, (self.input_size,))
        self.assertFalse(np.array_equal(self.layer.weights, old_weights))
        self.assertFalse(np.array_equal(self.layer.bias, old_bias))

    def test_call_method(self):
        """Test __call__ method"""
        input_vec = np.array([1., 2., 3.])
        forward_output = self.layer.forward(input_vec)
        call_output = self.layer(input_vec)
        self.assertTrue(np.array_equal(forward_output, call_output))

    def test_custom_learning_rate(self):
        """Test custom learning rate in backpropagation"""
        self.layer.activateTraining()
        input_vec = np.array([1., 2., 3.])
        self.layer.forward(input_vec)
        dloss = np.array([0.1, 0.2])
        
        weights_before = self.layer.weights.copy()
        self.layer.backpropagate(dloss, learningRate=0.1)
        weights_after_custom = self.layer.weights.copy()
        
        self.assertFalse(np.array_equal(weights_before, weights_after_custom))


if __name__ == '__main__':
    unittest.main()