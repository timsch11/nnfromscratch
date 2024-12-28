import unittest
import numpy as np
from nn.neuralNetwork import NN
from nn.linlayer import linlayer
from nn.layer import layer


class TestNN(unittest.TestCase):
    def setUp(self):
        # Create sample layers
        self.layer1 = linlayer("relu", 3, 4)
        self.layer2 = linlayer("relu", 4, 2)
        self.nn = NN("l2", self.layer1, self.layer2)
        
        # Sample data
        self.X = np.array([1.0, 2.0, 3.0])
        self.y = np.array([0.0, 1.0])

    def test_initialization(self):
        """Test neural network initialization"""
        self.assertEqual(len(self.nn.layers), 2)
        self.assertTrue(callable(self.nn.lossFunction))
        self.assertTrue(callable(self.nn.lossFunctionDerivative))

    def test_invalid_loss_function(self):
        """Test initialization with invalid loss function"""
        with self.assertRaises(KeyError):
            NN("invalid_loss", self.layer1, self.layer2)

    def test_training_mode(self):
        """Test training mode activation/deactivation"""
        self.nn.train(self.X, self.y)
        
        # Check all layers are back in inference mode after training
        for layer in self.nn.layers:
            self.assertFalse(layer.training)

    def test_prediction_shape(self):
        """Test prediction output shape"""
        output = self.nn.predict(self.X)
        self.assertEqual(output.shape, (2,))  # Should match last layer's output size

    def test_prediction_normalization(self):
        """Test if prediction normalizes input correctly"""
        X_unnormalized = np.array([10.0, 20.0, 30.0])
        output = self.nn.predict(X_unnormalized)
        self.assertTrue(np.all(output >= 0))  # ReLU output should be non-negative

    def test_training_with_different_learning_rates(self):
        """Test training with different learning rates"""
        # Store initial weights
        initial_weights1 = self.layer1.weights.copy()
        initial_weights2 = self.layer2.weights.copy()

        # Train with different learning rates
        self.nn.train(self.X, self.y, learningRate=0.01)
        weights_small_lr1 = self.layer1.weights.copy()
        weights_small_lr2 = self.layer2.weights.copy()

        # Reset weights
        self.layer1.weights = initial_weights1.copy()
        self.layer2.weights = initial_weights2.copy()

        self.nn.train(self.X, self.y, learningRate=0.1)
        weights_large_lr1 = self.layer1.weights.copy()
        weights_large_lr2 = self.layer2.weights.copy()

        # Changes should be different for different learning rates
        self.assertFalse(np.allclose(weights_small_lr1, weights_large_lr1))
        self.assertFalse(np.allclose(weights_small_lr2, weights_large_lr2))

    def test_prediction_consistency(self):
        """Test if predictions are consistent for same input"""
        pred1 = self.nn.predict(self.X)
        pred2 = self.nn.predict(self.X)
        self.assertTrue(np.array_equal(pred1, pred2))

    def test_training_updates_weights(self):
        """Test if training updates weights"""
        initial_weights = [layer.weights.copy() for layer in self.nn.layers]
        self.nn.train(self.X, self.y)
        final_weights = [layer.weights.copy() for layer in self.nn.layers]
        
        for init_w, final_w in zip(initial_weights, final_weights):
            self.assertFalse(np.array_equal(init_w, final_w))

    def test_multi_sample_prediction(self):
        """Test prediction with multiple samples"""
        X_multi = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with self.assertRaises(ValueError):
            # Should raise error as current implementation expects 1D input
            self.nn.predict(X_multi)


if __name__ == '__main__':
    unittest.main()