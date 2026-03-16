#A. Z-Score Method
import pandas as pd
from scipy.stats import zscore # Salary data
df = pd.DataFrame({
    'Employee_ID': ['E01','E02','E03','E04','E05','E06','E07'],
    'Salary': [48000, 52000, 50500, 49200, 51000, 120000, 18000]
})
           # Z-score calculation
df['Z_Score'] = zscore(df['Salary'])
print(df['Z_Score'])
           # Detect outliers
outliers = df[abs(df['Z_Score']) > 2]
print(outliers)


#B. IQR Method
import pandas as pd
# Employee salary data
data = {
    'Employee_ID': ['E01','E02','E03','E04','E05','E06','E07'],
    'Salary': [48000, 52000, 50500, 49200, 51000, 120000, 18000]
}
df = pd.DataFrame(data)

# Calculate Q1 and Q3
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[
    (df['Salary'] < lower_bound) |
    (df['Salary'] > upper_bound)
]
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("\nDetected Outliers:")
print(outliers)




'''#AIM: Anomaly detection
1.An organization wants to identify abnormal employee salary values that deviate significantly from the normal salary range due to data entry errors or exceptional cases. Apply the statistical method (Z-Score/IQR) for the same.
Employee_ID	Salary (₹)
E01	48,000
E02	52,000
E03	50,500
E04	49,200
E05	51,000
E06	120,000
E07	18,000'''
