from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from silhouettePlot import silhouette_plot
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# generate sample-clustered data - remember, that real-world data should be standardized, to avoid different 'weights' for different dimensions/features
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

# show generated data
plt.figure(num='Generated sample-clustered data')
plt.scatter(X[:, 0], X[:, 1],
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()


# K-means clustering using scikit-learn
km = KMeans(n_clusters=3,  # the drawback of k-clustering - we have to set the number of clusters
            init='random',  # simple K-means (randomly inited centroids from sample data), later we use K-means++ (inited centroids far away from each other), which is the DEFAULT value of KMeans() and a smarter one
            n_init=10,  # how many trials, from which KMeans obtain the best result
            max_iter=300,  # max iterations (if tol is not achieved)
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# show fitted results
plt.figure(num='Clustered results (k-means)')
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            s=50, c='orange', marker='o', edgecolor='black', label='cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],
            s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, marker='*', c='red', edgecolor='black', label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

'''Hard versus soft clustering:
Hard clustering describes a family of algorithms where each sample in a dataset
is assigned to exactly one cluster, as in the k-means algorithm. 
In contrast, algorithms for soft clustering (sometimes also
called fuzzy clustering) assign a sample to one or more clusters. A popular example
of soft clustering is the fuzzy C-means (FCM) algorithm (also called soft k-means or
fuzzy k-means). As a result, we obtain not binary-membership of samples to clusters, but 
probabilities that a sample is a member of any cluster. For 2022 not, FCM is not implemented in scikit-learn.
'''


# Clustering evaluation #1: the elbow method to find the optimal number of clusters
'''
In 'inertia_' parameter of KMeans() there is the computed SumSquaredError between sample positions and computed 
clusters' centers. If there are (for a fixed dataset) more clusters k in KMeans, the SSE will decrease (it's obvious).
Elbow method idea is to identify the 'k' value where the SSE begins to increase MOST RAPIDLY. In plot
there would be an 'elbow' point, which is the optimal one.
'''
iP.print('Distortion: %.2f' % km.inertia_)  # (1) show distortion (SSE)

# the elbow method
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.figure(num='The elbow method (elbow point is the optimal number of clusters)')
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion (SSE)')
plt.tight_layout()
plt.show()


# Clustering evaluation #2: quantifying the quality of clustering  via silhouette plots
'''
1. Calculate the cluster cohesion a_i as the average distance between a sample x_i and all other points in the same cluster.
2. Calculate the cluster separation b_i from the next closest cluster as the average distance between the sample x_i and all samples in the nearest cluster.
3. Calculate the silhouette s_i as the difference between cluster cohesion and separation divided by the greater of the two: s_i = (b_i - a_i)/max(a_i,b_i)

The silhouette coefficient is bounded in the range -1 to 1. Based on the preceding
equation, it can be seen that the silhouette coefficient is 0 if the cluster separation
and cohesion are equal (b_i = a_i). Furthermore, we get close to an ideal silhouette
coefficient of 1 if b_i >> a_i, since b_i quantifies how dissimilar a sample is to other
clusters, and a_i tells us how similar it is to the other samples in its own cluster.
'''
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# generate silhouette plot (good)
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
silhouette_plot(y_km, silhouette_vals, title='Silhouette plot - \"good\" clustering')


# Comparison to "bad" clustering:
km = KMeans(n_clusters=2,  # 2 not 3 for "bad" result
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# show "badly" clustered result
plt.figure(num='Clustered results (BAD number of clusters [2])')
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            s=50, c='lightgreen', edgecolor='black', marker='s', label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            s=50, c='orange', edgecolor='black', marker='o', label='cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, marker='*', c='red', label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# generate silhouette plot (bad)
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
silhouette_plot(y_km, silhouette_vals, title='Silhouette plot - \"bad\" clustering')