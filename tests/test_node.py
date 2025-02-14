import unittest
from sigil.node import Node


class TNode(Node):
    def __init__(self, name):
        super().__init__(name)

    def compute(self):
        super().compute()

    def backward(self, upstream_gradient=1):
        super().backward()


class TestNode(unittest.TestCase):
    def test_initialization(self):
        node = TNode(name='MyNode')
        self.assertEqual(node.name, "MyNode")
        self.assertIsNone(node.value)


if __name__ == "__main__":
    unittest.main()
