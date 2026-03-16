#Q2.Logistic Regression
#Code :
#Step 01 – Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Step 02 – Load dataset
df = sns.load_dataset("titanic")

#Step 03 - Select relevant columns
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]]

#Step 04 – Handle missing values correctly (NO FutureWarning)
df["age"] = df["age"].fillna(df["age"].median())

#Step 05 – Convert categorical 'sex' to numeric
df["sex"] = df["sex"].map({"male": 0, "female": 1})

#Step 06 – Split features and target
X = df.drop("survived", axis=1)
y = df["survived"]

#Step 07 – Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Step 08 – Build and fit logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

#Step 09 – Predictions
y_pred = model.predict(X_test)

#Step 10 – Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

#Step 10 – Print results
print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

#Step 12 – Confusion Matrix Diagram
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
xticklabels=["Predicted 0", "Predicted 1"],
yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
