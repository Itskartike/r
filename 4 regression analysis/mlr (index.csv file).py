# B. Multiple Linear Regression (index.csv file)

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scipy.stats as stats

# Step 2: Load Data
house = pd.read_csv(r"D:\Sujal\bin\dsf\4 regression analysis\index.csv")

print(house.head())
print(house.shape)
print(house.info())
print(house.describe().T)
print(house.isnull().sum())

# Step 3: Pairplot
sns.pairplot(house[['death_rate','doctor_avail','hosp_avail',
                    'annual_income','density_per_capita']])
plt.suptitle("Pairwise plots", y=1.02)
plt.show()

# Step 4: Correlation Matrix
corr = house[['death_rate','doctor_avail','hosp_avail',
              'annual_income','density_per_capita']].corr()

print(corr)
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 5: Create Model
X = house[['death_rate','doctor_avail','hosp_avail','annual_income']]
y = house['density_per_capita']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Intercept (sklearn):", lr.intercept_)
print("Coefficients (sklearn):")
for name, coef in zip(X.columns, lr.coef_):
    print(f"{name}: {coef:.6f}")

# Step 6: OLS Summary (on training data)
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())

# Step 7: Predictions
y_test_pred = lr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print("Test RMSE:", rmse)
print("Test R2:", r2)

# Add predicted column (for full data visualization)
house['predicted'] = lr.predict(X)

# Step 8: Visualization

# (A) Scatter plots
for col in X.columns:
    plt.figure()
    sns.scatterplot(x=house[col], y=house['density_per_capita'])
    sns.regplot(x=house[col], y=house['density_per_capita'],
                ci=None, scatter=False, color='red')
    plt.xlabel(col)
    plt.ylabel('density_per_capita')
    plt.title(f'density_per_capita vs {col}')
    plt.show()

# (B) Actual vs Predicted
plt.figure()
sns.scatterplot(x=house['predicted'], y=house['density_per_capita'])
plt.plot([house['predicted'].min(), house['predicted'].max()],
         [house['predicted'].min(), house['predicted'].max()],
         color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Actual vs Predicted')
plt.show()

# (C) Residual Plot
residuals = house['density_per_capita'] - house['predicted']

plt.figure()
sns.scatterplot(x=house['predicted'], y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# (D) Q-Q Plot
sm.qqplot(residuals, line='45')
plt.title('Q-Q plot of residuals')
plt.show()

# Shapiro Test
stat, pval = stats.shapiro(residuals)
print("Shapiro-Wilk: stat=%.4f, p=%.4f" % (stat, pval))

# (E) Influence Plot
sm.graphics.influence_plot(model_sm, criterion="cooks")
plt.show()
