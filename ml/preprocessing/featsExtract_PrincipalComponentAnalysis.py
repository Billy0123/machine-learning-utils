import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from ml.utils.plotDecisionRegions import plot_decision_regions
from manual_FE_PCA import FE_PCA
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# import and standardize the wine data
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)
# if the Wine dataset is temporarily unavailable from the UCI machine learning repository, un-comment the following line of code to load the dataset from a local path:
# df_wine = pd.read_csv('../databases/wine.data', header=None)
# also, various datasets are available at: https://archive.ics.uci.edu/ml/datasets.php
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                     test_size=0.3, #30% test, 70% training (common: 60/40, 70/30, 80/20, but when big data set, even 90:10 or 99:1)
                     random_state=0,
                     stratify=y) #stratify: get the same proportions (approx) of various classes (y) in train/test sets
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test) # transforming with 'stdsc' fitted to the train data!


# Unsupervised dimensionality reduction via manual Principal Component Analysis
fePCA = FE_PCA(k_feats=3) # feats plot below supports k_feats=2 and 3
fePCA.fit(X_train_std)
X_train_pca = fePCA.transform(X_train_std)
iP.print('Eigenvalues \n%s' % fePCA.eigen_vals) # (1)
iP.print(('Transformation matrix W:\n', fePCA.w)) # (2)

# total and explained variance
tot = sum(fePCA.eigen_vals)
var_exp = [(i / tot) for i in sorted(fePCA.eigen_vals, reverse=True)]
iP.print(var_exp) # (3) explained (lambda_i/sum(lambda_i)) by manual PCA
cum_var_exp = np.cumsum(var_exp)

# plot explained variances (individual & cumulative) of each feature
plt.figure(num='Explained variances (individual & cumulative) of each feature by manual PCA')
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('fe_manual_PCA eigenvalues.png', dpi=300)
plt.show()

# feats plot
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
fig = plt.figure(num='Reduced feature space by manual PCA')
plt3D = fig.add_subplot(projection='3d')
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt3D.scatter(X_train_pca[y_train == l, 0],
                  X_train_pca[y_train == l, 1],
                  np.zeros(len(X_train_pca[y_train == l])) if fePCA.k_feats == 2 else X_train_pca[y_train == l, 2],
                  c=c, label=l, marker=m)
plt3D.set_xlabel('PC 1')
plt3D.set_ylabel('PC 2')
plt3D.set_zlabel('PC 3')
plt3D.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('fe_manual_PCA reduced-feature-space.png', dpi=300)
plt.show()


# Unsupervised dimensionality reduction via scikit-learn's Principal Component Analysis
pca = PCA() # n_components=None by default, so all features will remain unchanged -> no dimensionality reduction, just sorted information
X_train_pca = pca.fit_transform(X_train_std)
iP.print(pca.explained_variance_ratio_) # (4) explained (lambda_i/sum(lambda_i)) by scikit's PCA

# plot explained variances (individual & cumulative) of each feature
plt.figure(num='Explained variances (individual & cumulative) of each feature by scikit\'s PCA')
plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

# scikit's PCA, dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std) # transform train set
X_test_pca = pca.transform(X_test_std) # transform test set based on train's fit!

# feats plot
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
fig = plt.figure(num='Reduced feature space by scikit\'s PCA')
plt3D = fig.add_subplot(projection='3d')
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt3D.scatter(X_train_pca[y_train == l, 0],
                  X_train_pca[y_train == l, 1],
                  np.zeros(len(X_train_pca[y_train == l])) if pca.n_components_ == 2 else X_train_pca[y_train == l, 2],
                  c=c, label=l, marker=m)
plt3D.set_xlabel('PC 1')
plt3D.set_ylabel('PC 2')
plt3D.set_zlabel('PC 3')
plt3D.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('fe_scikit_PCA reduced-feature-space.png', dpi=300)
plt.show()

# Training logistic regression classifier using the first n_components principal components.
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

# plot classification results of log-reg using reduced features set for train data set
plot_decision_regions(X_train_pca, y_train, classifier=lr, title='Log-reg classification using reduced (scikit PCA) features set - train data set')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('fe_scikit_PCA log-reg classification results train data set.png', dpi=300)
plt.show()

# plot classification results of log-reg using reduced features set for test data set
plot_decision_regions(X_test_pca, y_test, classifier=lr, title='Log-reg classification using reduced (scikit PCA) features set - test data set')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('fe_scikit_PCA log-reg classification results test data set.png', dpi=300)
plt.show()