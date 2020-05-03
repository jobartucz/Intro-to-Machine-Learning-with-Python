import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from IPython.display import display

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

# let's make a up a flower with sepal length 5, sepal width 2.9, petal length 1, petal width 0.2
# and try to figure out what it is
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: ", X_new.shape)

prediction = knn.predict(X_new)
print("Prediction: ", prediction)
print("Predicted target name: ", iris_dataset['target_names'][prediction])

# evaluate the predictions from the test set
y_pred = knn.predict(X_test)
print ("Test set predictions:\n", y_pred)

print("Test set score: {:.2f".format(np.mean(y_pred == y_test)))
