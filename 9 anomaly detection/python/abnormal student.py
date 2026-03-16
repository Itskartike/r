import pandas as pd
from scipy.stats import zscore

# Student data
df = pd.DataFrame({
    'Student_ID': ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07'],
    'Marks': [72, 75, 78, 74, 76, 98, 35]
})
# Z-score calculation
df['Z_Score'] = zscore(df['Marks'])
# Print Z-scores
print("Z-Scores:")
print(df[['Student_ID', 'Marks', 'Z_Score']])
# Detect outliers
outliers = df[abs(df['Z_Score']) > 2]  # Use a threshold of 2 for identifying outliers
print("\nOutliers:")
print(outliers)

#IQR Method
import pandas as pd
# Student marks data
data = {
 'Student_ID': ['S01','S02','S03','S04','S05','S06','S07'],
 'Marks': [72, 75, 78, 74, 76, 98, 35]
}
df = pd.DataFrame(data)
# Calculate Q1 and Q3
Q1 = df['Marks'].quantile(0.25)
Q3 = df['Marks'].quantile(0.75)
print("Q1:", Q1)
print("Q3:", Q3)
# Calculate IQR
IQR = Q3 - Q1
print("IQR :", IQR)
# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Detect outliers
outliers = df[
 (df['Marks'] < lower_bound) |
 (df['Marks'] > upper_bound)
 ]
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("\nDetected Outliers:")
print(outliers)


'''
A university wants to identify abnormal student exam scores that deviate significantly from the normal score range due to evaluation errors or exceptional performance. Apply the statistical method (Z-Score/IQR) for the same.
Student_ID	Marks
S01	72
S02	75
S03	78
S04	74
S05	76
S06	98
S07	35
'''
