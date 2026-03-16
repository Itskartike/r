#Q3.Perform linear regression on the following data which refers to years of service in a factory of seven workers in a specialized field & their monthly income (in thousands of Rs.)
#Code :
#Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#Step 2: Load and prepare the data ( Data for 7 workers)
years = np.array([11,7,9,5,8,6,10])
income = np.array([17,15,13,12,16,14,18])
df = pd.DataFrame({"years": years, "income": income})

#Step 3: Visualization of Data
plt.figure()
sns.scatterplot(x="years", y="income", data=df, s=80)
sns.regplot(x="years", y="income", data=df, ci=None)
plt.title("Years of Service vs Monthly Income")
plt.xlabel("Years of Service")
plt.ylabel("Monthly Income (thousand Rs.)")
plt.show()

#Step 4 : Check the outliers
plt.figure()
sns.boxplot(y=df["income"])
plt.title("Boxplot of Monthly Income")
plt.show()

#Step 5 : Build the Linear Regression Model
X = df[["years"]]
y = df["income"]
lr = LinearRegression()
lr.fit(X, y)
print("Intercept:", lr.intercept_)
print("Slope:", lr.coef_[0])

#Step 6 : Detailed Model Summary using Statsmodel OLS
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

#Step 7 : Residuals
df["predicted"] = lr.predict(X)
df["residuals"] = df["income"] - df["predicted"]

#Step 8 : Residuals vs Fitted
plt.figure()
plt.scatter(df["predicted"], df["residuals"], s=60)
plt.axhline(0, linestyle='--')
plt.title("Residuals vs Fitted")
plt.show()

#Step 9 : Q-Q Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sm.qqplot(df["residuals"], line='45', fit=True, ax=ax)
plt.title("Q-Q Plot")
plt.show()

#Step 10: Histogram of residuals
plt.figure()
sns.histplot(df["residuals"], kde=True)
plt.title("Histogram of Residuals")
plt.show()
