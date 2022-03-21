import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Load iris training dataset
file = pd.read_csv(r"ddd.csv")

# feature of the dataset
X = file.iloc[:,0:16]
# target of the dataset
Y = file.iloc[:,16:17]

# split dataset into training dataset and test dataset
x_train ,x_test ,y_train , y_test = train_test_split(X,Y)

# # PLot data points
# plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
# plt.show()

# Create object of type MLPClassifier to access the needed functions
neuralNetworkClassifier = MLPClassifier(hidden_layer_sizes = (150,200), max_iter = 1000, solver='lbfgs', shuffle = True
                                        , learning_rate_init = 0.2)

# Solver 
# lbfgs’ is an optimizer in the family of quasi-Newton methods.
# ‘sgd’ refers to stochastic gradient descent.
# ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

# shuffle = true : to shuffle dataset

# Pass the training dataset and its corresponding targets(classes)
neuralNetworkClassifier.fit(x_train, y_train)

# Predict the targets of the test dataset
y_pred = neuralNetworkClassifier.predict(x_test)

# Compare the true targets with the predicted targets
# by computing the accuracy
print(accuracy_score(y_test, y_pred))
