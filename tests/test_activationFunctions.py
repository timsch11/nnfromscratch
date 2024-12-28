import unittest
import numpy as np
from nn.activationFunctions import relu, drelu, sigmoid, tanh, softmax


class TestActivationFunctions(unittest.TestCase):

    def test_relu(self):
        self.assertTrue(np.array_equal(relu(np.array([-1, 0, 1, 2])), np.array([0, 0, 1, 2])))
        self.assertTrue(np.array_equal(relu(-1), np.array([0])))
        self.assertTrue(np.array_equal(relu(2), np.array([2])))
        self.assertTrue(np.array_equal(relu([2, 3]), np.array([2, 3])))

    def test_drelu(self):
        self.assertTrue(np.array_equal(drelu(np.array([-1, 0, 1, 2])), np.array([0, 0, 1, 1])))
        self.assertTrue(np.array_equal(drelu(-1), np.array([0])))
        self.assertTrue(np.array_equal(drelu(2), np.array([1])))
        self.assertTrue(np.array_equal(drelu([2]), np.array([1])))

    def test_sigmoid(self):
        self.assertTrue(np.allclose(sigmoid(np.array([-1, 0, 1, 2])), np.array([0.26894142, 0.5, 0.73105858, 0.88079708])))
        self.assertTrue(np.allclose(sigmoid(-1), np.array([0.26894142])))
        self.assertTrue(np.allclose(sigmoid(2), np.array([0.88079708])))

    def test_tanh(self):
        self.assertTrue(np.allclose(tanh(np.array([-1, 0, 1, 2])), np.array([-0.76159416, 0, 0.76159416, 0.96402758])))
        self.assertTrue(np.allclose(tanh(-1), np.array([-0.76159416])))
        self.assertTrue(np.allclose(tanh(2), np.array([0.96402758])))

    def test_softmax(self):
        self.assertTrue(np.allclose(softmax(np.array([1, 2, 3])), np.array([0.09003057, 0.24472847, 0.66524096])))
        self.assertTrue(np.allclose(softmax(np.array([1, 2])), np.array([0.26894142, 0.73105858])))


if __name__ == '__main__':
    unittest.main()