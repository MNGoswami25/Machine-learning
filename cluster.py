#K-means clustering is a popular unsupervised machine learning algorithm used for partitioning a set of data points into a specified number of clusters.
# The goal of this algorithm is to group similar data points together and separate dissimilar ones

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # sklearn  It provides simple and efficient tools for data mining and data analysis.
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()

#X is a numerical matrix where each row corresponds to an iris flower, and the columns contain the measured feature values.
X = iris.data

# Specify the number of clusters (k)
num_clusters = 3

# Initialize the KMeans model
# Initialize the KMeans model with n_init explicitly set to 10
# This parameter controls the number of time the K-means algorithm will be run with different centroid seeds. The algorithm will start from different initial centroids
# and then choose the run that results in the lowest "inertia"
kmeans = KMeans(n_clusters=num_clusters, n_init=10)


# Fit the model to the data
kmeans.fit(X)

# Get the cluster assignments for each data point
labels = kmeans.labels_

# Get the cluster centers
#cluster_centers typically refers to the centroids of the clusters that are computed during the K-means clustering process.
cluster_centers = kmeans.cluster_centers_

# Visualize the clusters using the first two features
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=200)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-Means Clustering on Iris Dataset')
plt.show()

#In clustering, the goal is to group similar data points together based on their feature similarities, without using any predefined labels