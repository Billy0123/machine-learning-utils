import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


# perceptron run
ppn = Perceptron(max_iter=50, eta0=0.1, random_state=1)
#ppn = SGDClassifier(loss='perceptron', eta0=0.1, max_iter=50, random_state=1) # alternative scikit implementation with stochastic gradient (optimize cost function based on individual samples)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
iP.print('Misclassified samples: %d' % (y_test != y_pred).sum())
iP.print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
iP.print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))


# plot results
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150), title='Perceptron results')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('perceptron.png', dpi=300)
plt.show()