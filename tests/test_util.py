import unittest
import numpy as np
from nn.util import normalize


class TestNormalize(unittest.TestCase):
    def test_basic_normalization(self):
        """Test basic array normalization"""
        input_array = np.array([1, 2, 3, 4, 5])
        normalized = normalize(input_array)
        self.assertTrue(np.array_equal(normalized, np.array([0, 0.25, 0.5, 0.75, 1])))

    def test_range_bounds(self):
        """Test if output is always between 0 and 1"""
        input_array = np.array([-10, 0, 5, 10, 15, 20])
        normalized = normalize(input_array)
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
        self.assertAlmostEqual(np.min(normalized), 0)
        self.assertAlmostEqual(np.max(normalized), 1)

    def test_negative_values(self):
        """Test normalization with negative values"""
        input_array = np.array([-2, -1, 0, 1, 2])
        normalized = normalize(input_array)
        self.assertTrue(np.array_equal(normalized, np.array([0, 0.25, 0.5, 0.75, 1])))

    def test_single_value(self):
        """Test normalization with single value array"""
        input_array = np.array([5])
        normalized = normalize(input_array)
        self.assertTrue(np.array_equal(normalized, np.array([0])))

    def test_zero_array(self):
        """Test normalization of zero array"""
        input_array = np.zeros(5)
        normalized = normalize(input_array)
        self.assertTrue(np.array_equal(normalized, np.zeros(5)))

    def test_identical_values(self):
        """Test normalization when all values are identical"""
        input_array = np.full(5, 7)
        normalized = normalize(input_array)
        self.assertTrue(np.array_equal(normalized, np.zeros(5)))

    def test_different_shapes(self):
        """Test normalization with different array shapes"""
        input_array_2d = np.array([[1, 2], [3, 4]])
        normalized = normalize(input_array_2d)
        self.assertEqual(normalized.shape, input_array_2d.shape)
        self.assertAlmostEqual(np.min(normalized), 0)
        self.assertAlmostEqual(np.max(normalized), 1)

    def test_list_input(self):
        """Test normalization with list input"""
        input_list = [1, 2, 3, 4, 5]
        normalized = normalize(np.array(input_list))
        self.assertTrue(np.array_equal(normalized, np.array([0, 0.25, 0.5, 0.75, 1])))

    def test_float_values(self):
        """Test normalization with float values"""
        input_array = np.array([1.5, 2.5, 3.5, 4.5])
        normalized = normalize(input_array)
        self.assertAlmostEqual(np.min(normalized), 0)
        self.assertAlmostEqual(np.max(normalized), 1)

    def test_empty_array(self):
        """Test normalization with empty array"""
        with self.assertRaises(ValueError):
            normalize(np.array([]))


if __name__ == '__main__':
    unittest.main()