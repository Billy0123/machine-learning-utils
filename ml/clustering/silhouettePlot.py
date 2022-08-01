import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.metrics import silhouette_score


def silhouette_plot(y_fitted, silhouette_vals, title='Figure'):
    cluster_labels = np.unique(y_fitted)
    n_clusters = cluster_labels.shape[0]
    y_axis_lower, y_axis_upper = 0, 0
    y_ticks = []
    plt.figure(num=title)
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_fitted == c]
        c_silhouette_vals.sort()
        y_axis_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_axis_lower, y_axis_upper), c_silhouette_vals,
                 height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_axis_lower + y_axis_upper) / 2.)
        y_axis_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)  # it can be computed even without all the upper code (if, e.g., we don't want to plot the graph) with:  silhouette_score(X, y_km, metric='euclidean')
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(y_ticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()