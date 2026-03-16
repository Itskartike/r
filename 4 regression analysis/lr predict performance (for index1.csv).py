# Q5.Perform linear regression to predict performance index

# Step1: import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scipy.stats as stats

# Step2: Load Dataset
data = pd.read_csv(r"D:\Sujal\bin\dsf\4 regression analysis\index1.csv")   
print(data.head())
print(data.info())
print(data.describe().T)

# Step3: Pairplot
sns.pairplot(data[['index','written','language','tech','gk']])
plt.suptitle("Pairwise Plots", y=1.02)
plt.show()

# Step4: Correlation Matrix
corr = data[['index','written','language','tech','gk']].corr()
print("\nCorrelation Matrix:\n", corr)

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step5: Define X and Y
X = data[['written', 'language', 'tech', 'gk']]
y = data['index']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("\nIntercept:", lr.intercept_)
print("Coefficients:")
for name, coef in zip(X.columns, lr.coef_):   # ✅ Fixed indentation
    print(f"{name}: {coef:.6f}")

# Step6: Statsmodels OLS (on training data)
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())

# Predictions
y_test_pred = lr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print("\nTest RMSE:", rmse)
print("Test R2:", r2)

# Add predicted values for full dataset (for plotting)
data["predicted"] = lr.predict(X)

# Step7: Scatter Plots for Each Feature
for col in X.columns:
    plt.figure()
    sns.scatterplot(x=data[col], y=data['index'])
    sns.regplot(x=data[col], y=data['index'], scatter=False, ci=None, color="red")
    plt.title(f'Performance Index vs {col}')
    plt.show()

# Step8: Actual vs Predicted Plot
plt.figure()
sns.scatterplot(x=data["predicted"], y=data["index"])
plt.plot(
    [data["predicted"].min(), data["predicted"].max()],
    [data["predicted"].min(), data["predicted"].max()],
    linestyle="--",
    color="red"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Actual vs Predicted")
plt.show()

# Step9: Residual Analysis
residuals = data['index'] - data['predicted']

plt.figure()
sns.scatterplot(x=data["predicted"], y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Fitted")
plt.show()

# Step10: Q-Q Plot
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot")
plt.show()

# Step11: Shapiro Test
stat, pval = stats.shapiro(residuals)
print("\nShapiro-Wilk Test: stat=%.4f, p=%.4f" % (stat, pval))
