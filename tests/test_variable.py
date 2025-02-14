import unittest
import numpy as np
from sigil.variable import Variable


class TestVariable(unittest.TestCase):
    def test_initialization(self):
        var = Variable("MyVariable", shape=(3, 3), dtype=np.float32)
        self.assertEqual(var.shape, (3, 3))
        self.assertEqual(var.dtype, np.float32)
        self.assertTrue(var.trainable)
        self.assertEqual(max(var.grad, 1e-10), 1)


if __name__ == "__main__":
    unittest.main()
