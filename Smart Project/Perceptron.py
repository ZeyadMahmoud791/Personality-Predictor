from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
# Load iris training dataset
file = pd.read_csv(r"ddd.csv")
# feature of the dataset
X = file.iloc[:,0:16]
# target of the dataset
Y = file.iloc[:,16:17]

# split dataset into training dataset and test dataset
x_train ,x_test ,y_train , y_test = train_test_split(X,Y)
y_train=y_train.values
y_train=y_train.flatten()
# Create object of type perceptronClassifier to access the needed functions
perceptronClassifier = Perceptron()

# Pass the training dataset and its corresponding targets(classes)
perceptronClassifier.fit(x_train, y_train)

# Predict the targets of the test dataset
y_pred = perceptronClassifier.predict(x_test)

# Compare the true targets with the predicted targets
# by computing the accuracy
print(accuracy_score(y_test, y_pred))
