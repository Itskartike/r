#Aim: Regression Analysis
#Code:
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

#Q4.Perform linear regression on the iris dataset of R for predicting sepal.length on sepal.width.
'''●Load the data and visualize
●Model building
●Model testing
●Inference/Prediction'''

#Code :
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

#Q5.Perform linear regression on the given data to estimate the predict the value of performance index based on writing skills, language skills, technical knowledge and general knowledge.
#Code : (for index1.csv)
#Step1: import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scipy.stats as stats

#Step2: Load Dataset
data = pd.read_csv(r"D\index1.csv")
print(data.head())
print(data.info())
print(data.describe().T)

#Step3: Pairplot
sns.pairplot(data[['index','written','language','tech','gk']])
plt.suptitle("Pairwise Plots", y=1.02)
plt.show()

#Step4: Correlation Matrix
corr = data[['index','written','language','tech','gk']].corr()
print("\nCorrelation Matrix:\n", corr)
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#Step 5: Define X and Y
X = data[['written', 'language', 'tech', 'gk']]
y = data['index']
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print("\nIntercept:", lr.intercept_)
print("Coefficients:")
for name, coef in zip(X.columns, lr.coef_):
print(f" {name}: {coef:.6f}")

#Step 6: Statsmodels OLS
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())
data["predicted"] = model_sm.predict(X_sm)
y_test_pred = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)
print("\nTest RMSE:", rmse)
print("Test R2:", r2)

#Step 7: Scatter Plots for Each Feature
for col in X.columns:
plt.figure()
sns.scatterplot(x=data[col], y=data['index'])
sns.regplot(x=data[col], y=data['index'], scatter=False, ci=None)
plt.title(f'Performance Index vs {col}')
plt.show()

#Step 8: Actual vs Predicted Plot
plt.figure()
sns.scatterplot(x=data["predicted"], y=data["index"])
plt.plot([data["predicted"].min(), data["predicted"].max()],
[data["predicted"].min(), data["predicted"].max()],
linestyle="--", color="red")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Actual vs Predicted")
plt.show()

#Step 9: Residual Analysis
residuals = data['index'] - data['predicted']
plt.figure()
sns.scatterplot(x=data["predicted"], y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Fitted")
plt.show()

#Step 10: Q-Q Plot
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot")
plt.show()

#Step 11: Shapiro Test
stat, pval = stats.shapiro(residuals)
print("\nShapiro-Wilk Test: stat=%.4f, p=%.4f" % (stat, pval))

#B.Multiple Linear Regression (index.csv file)
#Code :

#Step 1: import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import scipy.stats as stats

#Step2: Load your data
house = pd.read_csv(r"D:\DS_PRACT\DS\mlr data\index.csv")
house.head()
house.shape()

#Step 3: Explore and understand the data (Pairplot)
house.info()
house.describe().T
house.columns
house.isnull().sum() # Check missing values
sns.pairplot(house[[‘death_rate’ , ‘doctor_avail’ , ‘hosp_avail’ , ‘annual_income’ , ‘density_per_capita’]])
plt.suptitle(“Pairwise plots ” , y = 1.02)
plt.show()

#Step 4 : Explore and understand the data (Correlation matrix)
corr = house[['death_rate','doctor_avail','hosp_avail','annual_income','density_per_capita']].corr()
print(corr)
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation matrix")
plt.show()

#Step4: Create the model
X = house[['death_rate','doctor_avail','hosp_avail','annual_income']]
y = house['density_per_capita']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Intercept (sklearn):", lr.intercept_)
print("Coefficients (sklearn):")
for name, coef in zip(X.columns, lr.coef_):
print(f" {name}: {coef:.6f}")

#Step5: Model summary
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

#Step6: Make predictions
house['predicted'] = model_sm.predict(X_sm)
y_test_pred = lr.predict(X_test)
new_obs = pd.DataFrame({'death_rate':[2.0], 'doctor_avail':[1.5], 'hosp_avail':[0.7], 'annual_income':[35000]})
pred_new = lr.predict(new_obs) # sklearn
print("Predicted density_per_capita for new_obs:", pred_new[0])
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)
print("Test RMSE:", rmse)
print("Test R2:", r2)

#Step7: Visualize the model
#(A) Scatter of each predictor vs response with regression line
for col in X.columns: # For each predictor, scatter + line
plt.figure()
sns.scatterplot(x=house[col], y=house['density_per_capita'])
sns.regplot(x=house[col], y=house['density_per_capita'], ci=None, scatter=False)
plt.xlabel(col)
plt.ylabel('density_per_capita')
plt.title(f'density_per_capita vs {col}')
plt.show()

#(B) Actual vs Predicted
plt.figure()
sns.scatterplot(x=house['predicted'], y=house['density_per_capita'])
plt.plot([house['predicted'].min(), house['predicted'].max()],
[house['predicted'].min(), house['predicted'].max()], color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Actual vs Predicted')
plt.show()

#(C) Residuals vs Fitted (diagnose heteroscedasticity / non-linearity)
residuals = house['density_per_capita'] - house['predicted']
plt.figure()
sns.scatterplot(x=house['predicted'], y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

#(D) Q–Q plot for residuals (normality)
sm.qqplot(residuals, line='45', fit=True)
plt.title('Q-Q plot of residuals')
plt.show()
# Shapiro test (numeric)
stat, pval = stats.shapiro(residuals)
print("Shapiro-Wilk: stat=%.4f, p=%.4f" % (stat, pval))

#(E) Influence / Leverage plot (outliers & influential points)
sm.graphics.influence_plot(model_sm, criterion="cooks")
plt.show()

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
Conclusion : The Titanic dataset was successfully loaded from Seaborn’s built-in dataset repository. It contains passenger information used to predict survival.
Step 03 - Select relevant columns
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
