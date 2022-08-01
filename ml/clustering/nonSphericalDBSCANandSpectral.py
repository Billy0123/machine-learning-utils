import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# generate and show sample-non-spherically-clustered data - remember, that real-world data should be standardized, to avoid different 'weights' for different dimensions/features
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.figure(num='Sample, non-spherically clustered data')
plt.scatter(X[:, 0], X[:, 1])
plt.tight_layout()
plt.show()


# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering:
'''
According to the DBSCAN algorithm, a special label is assigned to each sample (point) using the following criteria:
  (1) A point is considered a "core point" if at least a specified number (MinPts) of neighboring points fall within the specified radius 'eps'.
  (2) A "border point" is a point that has fewer neighbors than MinPts within 'eps', but lies within the 'eps' radius of a core point.
  (3) All other points that are neither core nor border points are considered "noise points".
After labeling the points as core, border, or noise, the DBSCAN algorithm can be summarized in two simple steps:
  (1) Form a separate cluster for each core point or connected group of core points (core points are connected if they are no farther away than 'eps').
  (2) Assign each border point to the cluster of its corresponding core point.
One of the main advantages of using DBSCAN is that it does not assume that the clusters have a spherical shape as in k-means. Furthermore, DBSCAN is different
from k-means and hierarchical clustering in that it doesn't necessarily assign each point to a cluster but is capable of removing noise points.
https://scikit-learn.org/stable/modules/clustering.html#dbscan
'''
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)


# Spectral clustering:
'''
In practice Spectral Clustering is very useful when the structure of the individual clusters is highly non-convex, or 
more generally when a measure of the center and spread of the cluster is not a suitable description of the complete 
cluster, such as when clusters are nested circles on the 2D plane. When calling fit, an affinity matrix is constructed 
using either a kernel function such the Gaussian (aka RBF) kernel with Euclidean distance d(X, X):
'np.exp(-gamma * d(X,X) ** 2)' or a k-nearest neighbors connectivity matrix.
https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
'''
sc = SpectralClustering(n_clusters=2, random_state=0, gamma=100)
y_sc = sc.fit_predict(X)


# K-means clustering:
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)


# Hierarchical clustering:
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
y_ac = ac.fit_predict(X)


# plot various clustering results
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), num='Comparison of various clustering algorithms')

ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            edgecolor='black', c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            edgecolor='black', c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means clustering')

ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1],
            c='lightblue', edgecolor='black', marker='o', s=40, label='cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1],
            c='red', edgecolor='black', marker='s', s=40, label='cluster 2')
ax2.set_title('Agglomerative clustering')

ax3.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
            c='lightblue', marker='o', s=40, edgecolor='black', label='cluster 1')
ax3.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
            c='red', marker='s', s=40, edgecolor='black', label='cluster 2')
ax3.set_title('DBSCAN clustering')

ax4.scatter(X[y_sc == 0, 0], X[y_sc == 0, 1],
            c='lightblue', marker='o', s=40, edgecolor='black', label='cluster 1')
ax4.scatter(X[y_sc == 1, 0], X[y_sc == 1, 1],
            c='red', marker='s', s=40, edgecolor='black', label='cluster 2')
ax4.set_title('Spectral clustering')

plt.legend()
plt.tight_layout()
plt.show()

# more clustering methods: https://scikit-learn.org/stable/modules/clustering.html