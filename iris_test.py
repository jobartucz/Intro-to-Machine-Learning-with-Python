import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from IPython.display import display

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()


# print("keys of iris_dataset:\n",iris_dataset.keys())

# print ("Target names:", iris_dataset['data'])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
print(iris_dataframe)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
