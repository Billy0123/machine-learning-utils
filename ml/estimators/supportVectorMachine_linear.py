from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from ml.utils.plotDecisionRegions import plot_decision_regions
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


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


# Support Vector Machine - linear (Support Vector Classification class) run
svm = SVC(kernel='linear', C=1.0, max_iter=50, random_state=1)
#svm = SGDClassifier(loss='hinge', n_iter=50, random_state=1) # alternative scikit implementation with stochastic gradient (optimize cost function based on individual samples)
svm.fit(X_train_std, y_train)


# plot results
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105, 150), title='SVM (linear) results')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('SVM-linear.png', dpi=300)
plt.show()