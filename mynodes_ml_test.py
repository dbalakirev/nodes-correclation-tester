import unittest
from mynodes_ml import MyNodesML

class TestMyNodesML(unittest.TestCase):

    def test_A(self):
        mynodes = MyNodesML()
        expected = 1
        result = mynodes.predict([[4,8,2,2,4,2]])
        self.assertEqual(expected, result)

    def test_B(self):
        mynodes = MyNodesML()
        expected = 1
        result = mynodes.predict([[4,8,3,1,4,2]])
        self.assertEqual(expected, result)

    def test_C(self):
        mynodes = MyNodesML()
        expected = 0
        result = mynodes.predict([[4,6,3,1,2,2]])
        self.assertEqual(expected, result)

if __name__ == '__main__':
    unittest.main()