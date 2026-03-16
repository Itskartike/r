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


'''
A fitness application wants to detect users whose daily activity behavior significantly deviates from that of similar users in their local group.Apply Local Outlier Factor (LOF)
User_ID	Steps_Per_Day	Workout_Sessions
U01	6,500	4
U02	7,000	5
U03	6,800	6
U04	7,200	5
U05	6,900	4
U06	18,000	15
U07	1,200	1
'''

