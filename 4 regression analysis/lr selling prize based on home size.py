#Q2.A sample of 10 homes sold in an area is selected and the following data was gathered, Perform linear regression to predict Selling prize based on home size. Predict the Selling prize for ●Loading the data and visualization
'''●Model building
●Model testing
●Inference/Prediction'''

#Code:
#Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats

#Step 2: Load and prepare the data
home_size = np.array([1400,1300,1200,950,900,1000,1300,850,1100])
selling_price = np.array([70,62,65,45,40,53,68,40,55])
df = pd.DataFrame({"home_size": home_size, "selling_price": selling_price})

#Step 3: Visualization of Data
plt.figure()
sns.scatterplot(x="home_size", y="selling_price", data=df, s=80)
sns.regplot(x="home_size", y="selling_price", data=df, ci=None, marker=None)
plt.title("home_size vs selling_price")
plt.xlabel("home_size")
plt.ylabel("selling_price")
plt.show()

#Step 4 : Check the outliers
plt.figure()
sns.boxplot(y=df["selling_price"])
plt.title("Boxplot of Selling Price")
plt.show()

#Step 5: Build the linear regression Model
X = df[["home_size"]]
y = df["selling_price"]
lr = LinearRegression()
lr.fit(X, y)
b0 = lr.intercept_ # coefficients
b1 = lr.coef_[0]
print(f"Intercept (b0): {b0:.4f}")
print(f"Slope (b1): {b1:.4f}")

#Step 6 : OLS Summary
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

#Step 7 : Prediction
new_size = pd.DataFrame({"home_size": [1500]})
predicted_price = lr.predict(new_size)
print("Predicted price for home size 1500:", predicted_price[0])

#Step 8 : Predicted and Residual Values
df["predicted"] = lr.predict(X)
df["residuals"] = df["selling_price"] - df["predicted"]

#Step 9 : Residuals vs Fitted Plots
plt.figure()
plt.scatter(df["predicted"], df["residuals"], s=60)
plt.axhline(0, linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

#Step 10 : Q-Q Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sm.qqplot(df["residuals"].values, line='45', fit=True, ax=ax)
plt.title("Q-Q Plot of Residuals")

plt.show()

#Step 11 : Histogram of Residuals
plt.figure()
sns.histplot(df["residuals"], kde=True)
plt.title("Histogram of Residuals")
plt.show()
