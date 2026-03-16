#Q 2) Create decision tree model on mtcars dataset.
#Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Step 2: Load the mtcars Dataset
mtcars = pd.read_csv(r"D:\Sujal\bin\dsf\2 classification using decision trees\mtcars.csv")
mtcars.head()

#Step 3: Define Features (X) and Target Variable (y)
mtcars = mtcars.drop(columns=['model'])
X = mtcars.drop('am', axis=1)
y = mtcars['am']

#Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Step 5: Create the Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion="gini",max_depth=3)

#Step 6: Train the Model
dt_model.fit(X_train, y_train)

#Step 7: Make Predictions on Test Data
y_pred = dt_model.predict(X_test)

#Step 8: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

#Step 9: Visualize the Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Automatic", "Manual"],
    filled=True
)
plt.show()

