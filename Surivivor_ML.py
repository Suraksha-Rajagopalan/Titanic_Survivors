import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn import tree

df = pd.read_csv("DataSet_Titanic.csv")

df.head()

X = df.drop('Sobreviviente', axis=1)
y = df.Sobreviviente

y.head()

my_tree = DecisionTreeClassifier(max_depth = 3, random_state = 42)

my_tree.fit(X, y)

prediction_y = my_tree.predict(X)
print("Accuracy: ", accuracy_score(prediction_y, y))

confusion_matrix(y, prediction_y)

plot_confusion_matrix(my_tree, X, y, cmap = plt.cm.Blues, values_format='.0f')

plot_confusion_matrix(my_tree, X, y, cmap = plt.cm.Blues, values_format='.0f', normalize='true')

plt.figure(figsize = (10, 8))
tree.plot_tree(my_tree, filled = True, feature_names = X.columns)
plt.show()

relevance = my_tree.feature_importances_
columns = X.columns

sns.barplot(columns, relevance)
plt.title("Relevance of each attribute")
plt.show()
