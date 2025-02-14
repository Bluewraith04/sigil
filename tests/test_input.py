import unittest
import numpy as np
from sigil.input import Input


class TestInput(unittest.TestCase):
    def test_set_value(self):
        input_node = Input(name="MyInput", expected_shape=(2,))
        value = np.array([[1, 2], [3, 4]])
        input_node.set_value(value=value)
        np.testing.assert_array_equal(input_node.value, value)

    def test_input_shape_mismatch(self):
        input_node = Input(name="MyInput", expected_shape=(2,))
        value = np.array([1, 2, 3])  # Shape mismatch
        with self.assertRaises(ValueError):  # Expect a ValueError
            input_node.set_value(value)
        