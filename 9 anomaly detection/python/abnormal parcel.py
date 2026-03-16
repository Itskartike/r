from sklearn.neighbors import NearestNeighbors
import pandas as pd
# Create DataFrame with Delivery_Time
df = pd.DataFrame({
 'Order_ID': ['O01', 'O02', 'O03', 'O04', 'O05', 'O06', 'O07'],
 'Delivery_Time': [46, 48, 50, 47, 49, 95, 10]
})
# Reshape the data to 2D array as required by NearestNeighbors
X = df[['Delivery_Time']].values
# Initialize the NearestNeighbors model and fit the data
nbrs = NearestNeighbors(n_neighbors=3)
nbrs.fit(X)
# Calculate distances between neighbors and the indices of the nearest neighbors
distances, _ = nbrs.kneighbors(X)
# Add a new column for the average distance to the nearest neighbors
df['Avg_Distance'] = distances.mean(axis=1)
# Debugging: Print the average distances to see if they're too small
print("Average Distances:\n", df[['Order_ID', 'Avg_Distance']])
# Define a threshold based on mean and standard deviation
threshold = df['Avg_Distance'].mean() + 1 * df['Avg_Distance'].std() # Reduced
threshold
# Debugging: Print the threshold
print(f"\nThreshold: {threshold}")
# Identify outliers by comparing average distance to the threshold
outliers = df[df['Avg_Distance'] > threshold]
# Display the outliers
print("\nOutliers:\n", outliers)


'''
A logistics company aims to identify abnormal parcel delivery times that differ significantly from nearby delivery records. Use (k-NN)
Order_ID	Delivery_Time (Hours)
O01	46
O02	48
O03	50
O04	47
O05	49
O06	95
O07	10
'''
