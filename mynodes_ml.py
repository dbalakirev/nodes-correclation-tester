import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class MyNodesML:

    def __init__(self):
        self.__maxprice = 900

        # read initial data
        dataset = pd.read_csv("data.csv")

        # input
        x_train = dataset.iloc[:, [1,2,3,4,5,6]].values
        # output
        y_train = dataset.iloc[:, 7].values
        
        self.__classifier = LogisticRegression(random_state = 0)
        self.__classifier.fit(x_train, y_train)

    def predict(self, x):
        prediction = self.__classifier.predict(x)
        return prediction
