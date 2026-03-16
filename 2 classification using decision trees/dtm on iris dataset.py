#Aim: Classification Using Decision Trees

#Q 1) Create decision tree model on iris dataset and predict the Species.                                                                                  
#Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Step 2: Load the Dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

#Step 3: Split the Dataset into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Step 4: Create the Decision Tree Classifier Model
dt_model = DecisionTreeClassifier(criterion="gini", max_depth=3)

#Step 5: Train (Fit) the Model
dt_model.fit(X_train, y_train)

#Step 6: Make Predictions on Test Data
y_pred = dt_model.predict(X_test)
print(y_pred)

#Step 7: Evaluate the Model Performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

#Step 8: Visualize the Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plot_tree(dt_model, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True)
plt.show()

