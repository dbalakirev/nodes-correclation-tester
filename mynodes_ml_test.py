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
            self.assertEqual(expected, result, "Code {} did not work!".format(code))

    # D is not added to the dataset yet
    def test_predict_unseen_record(self):
        x = [[4,8,2,2,4,3]]
        expected = 0
        mynodes = MyNodesML()
        result = mynodes.predict(x)
        self.assertEqual(expected, result, "Example D {} did not work!".format(x))

if __name__ == '__main__':
    unittest.main()