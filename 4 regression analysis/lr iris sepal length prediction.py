#Q4.Perform linear regression on the iris dataset of R for predicting sepal.length on sepal.width.
'''●Load the data and visualize
●Model building
●Model testing
●Inference/Prediction'''


#Step 1 : Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#Step 2: Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.rename(columns={
"sepal length (cm)": "sepal_length",
"sepal width (cm)": "sepal_width",
"petal length (cm)": "petal_length",
"petal width (cm)": "petal_width"
}, inplace=True)
print(df.head())

#Step 3: Visualization (Scatterplot)
plt.figure()
sns.scatterplot(x="sepal_width", y="sepal_length", data=df, s=80)
sns.regplot(x="sepal_width", y="sepal_length", data=df, ci=None)
plt.title("Sepal Width vs Sepal Length")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Sepal Length (cm)")
plt.show()

#Step 4 : Check the outliers
plt.figure()
sns.boxplot(y=df["sepal_length"])
plt.title("Boxplot of Sepal Length")
plt.show()

#Step 5: Build the Model
X = df[["sepal_width"]]
y = df["sepal_length"]
lr = LinearRegression()
lr.fit(X, y)
print("Intercept:", lr.intercept_)
print("Slope:", lr.coef_[0])


#Step 6 : Statsmodels Detailed Summary
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

#Step 7: Residual Analysis
df["predicted"] = lr.predict(X)
df["residuals"] = df["sepal_length"] - df["predicted"]

#Step 8 : Residual vs Fitted
plt.figure()
plt.scatter(df["predicted"], df["residuals"], s=60)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Fitted Values")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

#Step 9 : Q-Q Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sm.qqplot(df["residuals"], line="45", fit=True, ax=ax)
plt.title("Q-Q Plot of Residuals")
plt.show()

#Step 10 : Histogram of Residuals
plt.figure()
sns.histplot(df["residuals"], kde=True)
plt.title("Histogram of Residuals")
plt.show()
