import numpy as np


# manual implementation of Principal Component Analysis class
class FE_PCA:
    def __init__(self, k_feats=2):
        self.k_feats=k_feats

    def fit(self, X):
        self.X = X

        # Eigendecomposition of the covariance matrix.
        self.cov_mat = np.cov(self.X.T)
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(self.cov_mat)

        # Feature transformation:
        self.eigen_pairs = [(np.abs(self.eigen_vals[i]), self.eigen_vecs[:, i])  # make a list of (eigenvalue, eigenvector) tuples
                            for i in range(len(self.eigen_vals))]
        self.eigen_pairs.sort(key=lambda k: k[0], reverse=True) # sort the (eigenvalue, eigenvector) tuples from high to low
        self.w = np. hstack([self.eigen_pairs[i][1][:, np.newaxis] for i in range(self.k_feats)])

    def transform(self, X):
        return X.dot(self.w)