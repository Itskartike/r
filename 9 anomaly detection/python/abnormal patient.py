from sklearn.neighbors import NearestNeighbors
import pandas as pd
# Create DataFrame with Waiting_Time
df = pd.DataFrame({
 'Patient_ID': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07'],
 'Waiting_Time': [28, 30, 32, 29, 31, 90, 5]
})
# Reshape the data to 2D array as required by NearestNeighbors
X = df[['Waiting_Time']].values
# Initialize the NearestNeighbors model and fit the data
nbrs = NearestNeighbors(n_neighbors=3)
nbrs.fit(X)
# Calculate distances between neighbors and the indices of the nearest neighbors
distances, _ = nbrs.kneighbors(X)
# Add a new column for the average distance to the nearest neighbors
df['Avg_Distance'] = distances.mean(axis=1)
# Debugging: Print the average distances to see if they're too small
print("Average Distances:\n", df[['Patient_ID', 'Avg_Distance']])
# Define a threshold based on mean and standard deviation
threshold = df['Avg_Distance'].mean() + 1 * df['Avg_Distance'].std() # Reduced threshold
# Debugging: Print the threshold
print(f"\nThreshold: {threshold}")
# Identify outliers by comparing average distance to the threshold
outliers = df[df['Avg_Distance'] > threshold]
# Display the outliers
print("\nOutliers:\n", outliers)

'''
A hospital wants to identify abnormal patient waiting times that differ significantly from other patients’ waiting times. Apply K-NN method.
Patient_ID	Waiting_Time (Minutes)
P01	28
P02	30
P03	32
P04	29
P05	31
P06	90
P07	5
'''
