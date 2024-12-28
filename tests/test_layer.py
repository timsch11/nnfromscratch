import unittest
import numpy as np
from nn.layer import layer


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.layer = layer()
        
    def test_initialization_default(self):
        """Test layer initialization with default parameters"""
        self.assertEqual(self.layer.learningRate, 0.01)
        self.assertFalse(self.layer.training)
        self.assertTrue(callable(self.layer.activation))
        
    def test_initialization_with_relu(self):
        """Test layer initialization with ReLU activation"""
        layer_relu = layer(activationFunction="relu")
        self.assertTrue(callable(layer_relu.activation))
        self.assertTrue(callable(layer_relu.derivativeActivation))
        
    def test_initialization_invalid_activation(self):
        """Test layer initialization with invalid activation function"""
        with self.assertRaises(KeyError):
            layer(activationFunction="invalid_function")
            
    def test_learning_rate(self):
        """Test custom learning rate"""
        custom_layer = layer(learningRate=0.001)
        self.assertEqual(custom_layer.learningRate, 0.001)
        
    def test_training_mode(self):
        """Test training mode activation/deactivation"""
        self.layer.activateTraining()
        self.assertTrue(self.layer.training)
        
        self.layer.deactivateTraining()
        self.assertFalse(self.layer.training)
        
    def test_kaiming_initialization(self):
        """Test Kaiming He initialization"""
        input_size, output_size = 100, 50
        weights = layer.kaimingHeInitialization(input_size, output_size)
        
        self.assertEqual(weights.shape, (input_size, output_size))
        self.assertAlmostEqual(np.mean(weights), 0, delta=0.1)
        self.assertAlmostEqual(np.std(weights), np.sqrt(2/input_size), delta=0.1)
        
    def test_xavier_initialization(self):
        """Test Xavier initialization"""
        input_size, output_size = 100, 50
        weights = layer.xavierInitialization(input_size, output_size)
        
        self.assertEqual(weights.shape, (input_size, output_size))
        self.assertAlmostEqual(np.mean(weights), 0, delta=0.1)
        self.assertAlmostEqual(np.std(weights), np.sqrt(1/input_size), delta=0.1)
        
    def test_initialize_weights(self):
        """Test default weight initialization"""
        input_size, output_size = 100, 50
        weights = layer.initializeWeights(input_size, output_size)
        
        self.assertEqual(weights.shape, (input_size, output_size))
        self.assertAlmostEqual(np.mean(weights), 0, delta=0.1)
        self.assertAlmostEqual(np.std(weights), 1, delta=0.1)


if __name__ == '__main__':
    unittest.main()