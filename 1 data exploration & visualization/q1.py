











































#Aim:data exploration and visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
df = sns.load_dataset('iris')
df
#Step 2: Display First & Last Rows
print(df.head())
print(df.tail())
#Step 3: Dataset Structure
df.info()
"""Conclusion & Findings:
- 150 entries
- No null values
- 4 float features + 1 categorical species column."""
#Step 4: Summary Statistics
df.describe()
#Step 5: Check Missing Values
df.isnull().sum()
#A) Histogram
df.hist(figsize=(10,6))
plt.show()
#B) Boxplot
sns.boxplot(data=df)
plt.show()
#C) Countplot (Species Distribution)
sns.countplot(x='species', data=df)
plt.show()
#D) Scatter Plot (Sepal Length vs Sepal Width)
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
#E) Pairplot
sns.pairplot(df, hue='species')
plt.show()
#F) Scatter Multiple
plt.figure(figsize=(10,6))
plt.scatter(df['petal_length'], df['sepal_length'], label='Sepal Length')
plt.scatter(df['petal_length'], df['sepal_width'], label='Sepal Width')
plt.scatter(df['petal_length'], df['petal_width'], label='Petal Width')

plt.xlabel('Petal Length')
plt.ylabel('Values of Other Attributes')
plt.title('Scatter Multiple Plot (Iris Dataset)')
plt.legend()
plt.show()
#F) Scatter Matrix
from pandas.plotting import scatter_matrix
plt.figure(figsize=(10,8))
scatter_matrix(df.iloc[:, :4], figsize=(10,8), diagonal='hist')
plt.show()
#G) Parallel Coordinates Plot

from pandas.plotting import parallel_coordinates

plt.figure(figsize=(10,6))
parallel_coordinates(df, 'species')
plt.title('Parallel Coordinates Plot - Iris Dataset')
plt.show()
#H) Deviation Chart (from Mean)
mean_vals = df.iloc[:, :4].mean()

plt.figure(figsize=(10,6))
plt.plot(df.iloc[:, :4] - mean_vals)
plt.title('Deviation Chart - Deviation from Mean')
plt.xlabel('Record Index')
plt.ylabel('Deviation')
plt.show()
#I) Andrews Curve

from pandas.plotting import andrews_curves
plt.figure(figsize=(10,6))
andrews_curves(df, 'species')
plt.title('Andrews Curves - Iris Dataset')
plt.show()















































