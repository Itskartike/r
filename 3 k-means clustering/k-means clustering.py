#K-Means Clustering on Iris Dataset
#Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

#Step 2: Load the Iris Dataset
iris = load_iris()
X = iris.data

#Step 3: Decide the Number of Clusters (K)
k = 3

#Step 4: Apply K-Means Algorithm
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

#Step 5: Obtain Cluster Labels and Centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print(labels)
print(centroids)

#Step 6: Visualize the Clusters
plt.scatter(X[:,2], X[:,3], c=labels)
plt.scatter(centroids[:,2], centroids[:,3], marker='X')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clustering on Iris Dataset")
plt.show()
 
