#DBSCAN Using Iris Dataset 
#Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

#Step 2: Load the real dataset
iris = load_iris()
X = iris.data

#Step 3: Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Step 4: Create DBSCAN model
dbscan = DBSCAN(eps=0.6, min_samples=5)

#Steps 5: Apply DBSCAN algorithm
labels = dbscan.fit_predict(X_scaled)

#Step 6: Analyze the clusters formed
print(np.unique(labels))

#Step 7: Visualize the clustering result
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("DBSCAN Clustering on Iris Dataset")
plt.show()
