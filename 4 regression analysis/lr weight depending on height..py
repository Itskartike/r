#Simple Linear Regression
#Perform linear regression on the following data to predict weight of the person depending on height. Also predict the weight of person whose height is 140. 

#Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

#Step 2: Load and prepare the data
height = np.array([151,174,138,186,128,136,179,163,152])
weight = np.array([63,81,56,91,47,57,76,72,62])
df = pd.DataFrame({"height": height, "weight": weight})

#Step 3: Visualization of Data
sns.scatterplot(x="height", y="weight", data=df, s=80)
sns.regplot(x="height", y="weight", data=df, ci=None, scatter=False)
plt.title("Height vs Weight")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

#Step 4: Check the outliers
plt.figure()
sns.boxplot(y=df["weight"])
plt.title("Boxplot of Weight")
plt.show()

#Step 5: Build the linear regression Model
X = df[["height"]] # 2D
y = df["weight"]
lr = LinearRegression()
lr.fit(X, y)
b0 = lr.intercept_# Coefficients
b1 = lr.coef_[0]
print(f"Intercept (b0): {b0:.4f}")
print(f"Slope (b1): {b1:.4f}")

#Step 6: OLS Summary
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

#Step7: Predict weight for height =140
new_height = pd.DataFrame({"height": [140]})
predicted_weight = lr.predict(new_height)
print("Predicted weight for height 140:", predicted_weight[0])

#Step 8: Analyze Residual
df["predicted"] = lr.predict(X)
df["residuals"] = df["weight"] - df["predicted"]
plt.figure()
plt.scatter(df["predicted"], df["residuals"], s=60)
plt.axhline(0, linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

#Step 9: check normality of residual (Q-Q plot)
plt.figure()
sm.qqplot(df["residuals"], line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()

#Step 10: Histogram of residuals
plt.figure()
sns.histplot(df["residuals"], kde=True)
plt.title("Histogram of Residuals")
plt.show()
