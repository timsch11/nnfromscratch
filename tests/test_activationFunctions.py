import unittest
import numpy as np
from nn.activationFunctions import relu, drelu, sigmoid, dsigmoid, tanh, dtanh, softmax, dsoftmax

class TestActivationFunctions(unittest.TestCase):

    def test_relu(self):
        self.assertTrue(np.array_equal(relu(np.array([-1, 0, 1])), np.array([0, 0, 1])))
        self.assertTrue(np.array_equal(relu(-1), np.array([0])))
        self.assertTrue(np.array_equal(relu(1), np.array([1])))

    def test_drelu(self):
        self.assertTrue(np.array_equal(drelu(np.array([-1, 0, 1]), np.array([0, 0, 1])), np.array([0, 0, 1])))
        self.assertTrue(np.array_equal(drelu(-1, np.array([0])), np.array([0])))
        self.assertTrue(np.array_equal(drelu(1, np.array([1])), np.array([1])))

    def test_sigmoid(self):
        self.assertTrue(np.allclose(sigmoid(np.array([-1, 0, 1])), np.array([0.26894142, 0.5, 0.73105858])))
        self.assertTrue(np.allclose(sigmoid(-1), np.array([0.26894142])))
        self.assertTrue(np.allclose(sigmoid(1), np.array([0.73105858])))

    def test_dsigmoid(self):
        postActivation = sigmoid(np.array([-1, 0, 1]))
        self.assertTrue(np.allclose(dsigmoid(np.array([-1, 0, 1]), postActivation), postActivation * (1 - postActivation)))
        postActivation = sigmoid(-1)
        self.assertTrue(np.allclose(dsigmoid(-1, postActivation), postActivation * (1 - postActivation)))
        postActivation = sigmoid(1)
        self.assertTrue(np.allclose(dsigmoid(1, postActivation), postActivation * (1 - postActivation)))

    def test_tanh(self):
        self.assertTrue(np.allclose(tanh(np.array([-1, 0, 1])), np.array([-0.76159416, 0, 0.76159416])))
        self.assertTrue(np.allclose(tanh(-1), np.array([-0.76159416])))
        self.assertTrue(np.allclose(tanh(1), np.array([0.76159416])))

    def test_dtanh(self):
        postActivation = tanh(np.array([-1, 0, 1]))
        self.assertTrue(np.allclose(dtanh(np.array([-1, 0, 1]), postActivation), 1 - postActivation**2))
        postActivation = tanh(-1)
        self.assertTrue(np.allclose(dtanh(-1, postActivation), 1 - postActivation**2))
        postActivation = tanh(1)
        self.assertTrue(np.allclose(dtanh(1, postActivation), 1 - postActivation**2))

    def test_softmax(self):
        self.assertTrue(np.allclose(softmax(np.array([1, 2, 3])), np.array([0.09003057, 0.24472847, 0.66524096])))
        self.assertTrue(np.allclose(softmax(np.array([1])), np.array([1.0])))


if __name__ == '__main__':
    unittest.main()