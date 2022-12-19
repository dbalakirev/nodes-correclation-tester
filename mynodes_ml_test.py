import unittest
import pandas as pd
from mynodes_ml import MyNodesML

class TestMyNodesML(unittest.TestCase):

    def test_predict_all_trained_records(self):
        dataset = pd.read_csv("data.csv")
        for index, row in dataset.iterrows():
            code = row[0]
            expected = row[-1]
            mynodes = MyNodesML()
            x = [row[1:-1]]
            result = mynodes.predict(x)
            self.assertEqual(expected, result, "{} did not work!".format(code))

if __name__ == '__main__':
    unittest.main()