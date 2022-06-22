from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from ml.utils.plotDecisionRegions import plot_decision_regions
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# generate random XOR data set (impossible for classification by linear methods)
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)


# Support Vector Machine - non-linear [Radial Basis Function] (Support Vector Classification class) run for non-linear data set
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)


# plot results
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                      classifier=svm, title='Kernel SVM (RBF) for non-linear data set')

plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('kSVM results for non-linear dataset.png', dpi=300)
plt.show()


# import data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


# split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
iP.print(('Labels counts in y:', np.bincount(y)))
iP.print(('Labels counts in y_train:', np.bincount(y_train)))
iP.print(('Labels counts in y_test:', np.bincount(y_test)))


# standardize data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# Support Vector Machine - non-linear [Radial Basis Function] (Support Vector Classification class) run for linear data set
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)


# plot results
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150), title='Kernel SVM (RBF) for linear data set')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('kSVM results for linear dataset', dpi=300)
plt.show()


# Support Vector Machine - non-linear [Radial Basis Function] (Support Vector Classification class) run for linear data set, drastically increase gamma (overfitting)
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)


#plot results
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150), title='Kernel SVM (RBF) for linear data set, big gamma (overfitting)')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('kSVM results for linear dataset - overfitting.png', dpi=300)
plt.show()