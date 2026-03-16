#AIM: Anomaly detection
1.	An organization wants to identify abnormal employee salary values that deviate significantly from the normal salary range due to data entry errors or exceptional cases. Apply the statistical method (Z-Score/IQR) for the same.
Employee_ID	Salary (₹)
E01	48,000
E02	52,000
E03	50,500
E04	49,200
E05	51,000
E06	120,000
E07	18,000

A.	Z-Score Method
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


B. IQR Method
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


2.	A bank wants to detect unusual transaction amounts that may indicate fraudulent activity without using labeled fraud data. Use IsolationForest method
Transaction_ID	Transaction_Amount (₹)
T01	2,800
T02	3,200
T03	3,500
T04	2,900
T05	3,100
T06	18,000
T07	50
Code:
from sklearn.ensemble import IsolationForest
import pandas as pd
df = pd.DataFrame({
    'Amount': [2800, 3200, 3500, 2900, 3100, 18000, 50]
})
model = IsolationForest(contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(df)
print(df)

3.	A logistics company aims to identify abnormal parcel delivery times that differ significantly from nearby delivery records. Use (k-NN)
Order_ID	Delivery_Time (Hours)
O01	46
O02	48
O03	50
O04	47
O05	49
O06	95
O07	10
Code:
from sklearn.neighbors import NearestNeighbors
import pandas as pd
df = pd.DataFrame({
    'Delivery_Time': [46, 48, 50, 47, 49, 95, 10]
})
nbrs = NearestNeighbors(n_neighbors=3)
nbrs.fit(df)
distances, _ = nbrs.kneighbors(df)
df['Avg_Distance'] = distances.mean(axis=1)
threshold = df['Avg_Distance'].mean() + 2*df['Avg_Distance'].std()
outliers = df[df['Avg_Distance'] > threshold]
print(outliers)


4.	An e-commerce company wants to detect customers whose purchasing behavior significantly deviates from that of their local neighborhood. Use Local Outlier Factor (LOF)
Customer_ID	Spending_Score	Purchase_Frequency
CU01	45	18
CU02	50	20
CU03	48	22
CU04	52	19
CU05	49	21
CU06	98	65
CU07	5	2
Code: 
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
df = pd.DataFrame({
    'Spending_Score': [45, 50, 48, 52, 49, 98, 5],
    'Purchase_Frequency': [18, 20, 22, 19, 21, 65, 2]
})
lof = LocalOutlierFactor(n_neighbors=3)
df['Outlier'] = lof.fit_predict(df)
outliers = df[df['Outlier'] == -1] 
print(outliers) 

5.	A university wants to identify abnormal student exam scores that deviate significantly from the normal score range due to evaluation errors or exceptional performance. Apply the statistical method (Z-Score/IQR) for the same.
Student_ID	Marks
S01	72
S02	75
S03	78
S04	74
S05	76
S06	98
S07	35

Code:
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


6.	An online shopping platform wants to detect unusual order amounts that may indicate suspicious purchasing behavior without using labeled fraud data. Apply Isolation Forest method for the same

Order_ID	Order_Amount (₹)
O101	1,200
O102	1,450
O103	1,300
O104	1,500
O105	1,400
O106	12,000
O107	80

Code:

import pandas as pd
from sklearn.ensemble import IsolationForest

# Order amount data
data = {
    'Order_ID': ['O101', 'O102', 'O103', 'O104', 'O105', 'O106', 'O107'],
    'Order_Amount': [1200, 1450, 1300, 1500, 1400, 12000, 80]
}

df = pd.DataFrame(data)

# Reshape data for Isolation Forest (expects a 2D array)
order_amounts = df[['Order_Amount']]

# Apply Isolation Forest
model = IsolationForest(contamination=0.2)  # 20% contamination rate assumed
df['Outlier'] = model.fit_predict(order_amounts)

# Convert 1 to 'Normal' and -1 to 'Outlier'
df['Outlier'] = df['Outlier'].map({1: 'Normal', -1: 'Outlier'})

# Print results
print(df)


7.	A hospital wants to identify abnormal patient waiting times that differ significantly from other patients’ waiting times. Apply K-NN method.

Patient_ID	Waiting_Time (Minutes)
P01	28
P02	30
P03	32
P04	29
P05	31
P06	90
P07	5
Code :
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Patient waiting time data
df_waiting = pd.DataFrame({
    'Patient_ID': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07'],
    'Waiting_Time': [28, 30, 32, 29, 31, 90, 5]
})

# Apply k-NN for outlier detection (3 neighbors)
nbrs = NearestNeighbors(n_neighbors=3)
nbrs.fit(df_waiting[['Waiting_Time']])

# Calculate distances and average distances to nearest neighbors
distances, _ = nbrs.kneighbors(df_waiting[['Waiting_Time']])
df_waiting['Avg_Distance'] = distances.mean(axis=1)

# Set the threshold for outlier detection: 2 standard deviations above the mean
threshold = df_waiting['Avg_Distance'].mean() + 2 * df_waiting['Avg_Distance'].std()

# Detect outliers
outliers_waiting = df_waiting[df_waiting['Avg_Distance'] > threshold]

print("Detected Outliers (Waiting Times):")
print(outliers_waiting)


8.	A fitness application wants to detect users whose daily activity behavior significantly deviates from that of similar users in their local group.Apply Local Outlier Factor (LOF)
User_ID	Steps_Per_Day	Workout_Sessions
U01	6,500	4
U02	7,000	5
U03	6,800	6
U04	7,200	5
U05	6,900	4
U06	18,000	15
U07	1,200	1

Code:

from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

# Fitness data
df_fitness = pd.DataFrame({
    'User_ID': ['U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07'],
    'Steps_Per_Day': [6500, 7000, 6800, 7200, 6900, 18000, 1200],
    'Workout_Sessions': [4, 5, 6, 5, 4, 15, 1]
})

# Apply LOF for outlier detection
lof_fitness = LocalOutlierFactor(n_neighbors=3)
df_fitness['Outlier'] = lof_fitness.fit_predict(df_fitness[['Steps_Per_Day', 'Workout_Sessions']])

# LOF labels outliers as -1 and inliers as 1, so we map them to readable labels
df_fitness['Outlier'] = df_fitness['Outlier'].map({1: 'Normal', -1: 'Outlier'})
# Print the detected outliers
print("Detected Outliers (Fitness):")
print(df_fitness[df_fitness['Outlier'] == 'Outlier'])

