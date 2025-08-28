
# imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class preprocessing:
    def __init__(self):
        # Load the dataset
        df = pd.read_csv('Iris.csv')
        # removing Id column
        df = df.drop(columns=["Id"])
        # Label Encoders
        # we use Label encoders to convert categorical value to numerical
        le = LabelEncoder()
        df["Species"] = le.fit_transform(df["Species"])
        X = df.drop(columns=["Species"])
        y = df["Species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test

# Model Training
# Logisitic Regression

class LogisiticRegressionModel(preprocessing):
    def model_training(self):
     lr = LogisticRegression()
     lr.fit(self.X_train, self.y_train)
     # Print metric to get performace of the model
     print("accuracy of logisitic regression is", lr.score(self.X_test, self.y_test)* 100)

class KnnModel(preprocessing):
    def model_training(self):
     kn = KNeighborsClassifier()
     kn.fit(self.X_train, self.y_train)
     # Print metric to get performace of the model
     print("accuracy of knn is", kn.score(self.X_test, self.y_test)* 100)

def main():
   print("Logisitic Regression value:")
   lr_model = LogisiticRegressionModel()
   lr_model.model_training()

   print("KNN value:")
   knn_model = KnnModel()
   knn_model.model_training()

if __name__ == "__main__":
    main()