from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from ml.utils.plotDecisionRegions import plot_decision_regions
from manual_logisticRegression import LogisticRegressionGD
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


# use only 2 class (1vRest not implemented in manual realization)
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]


# manual log-reg run
lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)


# plot results
plot_decision_regions(X=X_train_01_subset,
                      y=y_train_01_subset,
                      classifier=lrgd, title='Manual log-reg results')

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('manual log-reg.png', dpi=300)
plt.show()


# scikit log-reg run
lr = LogisticRegression(C=100.0, max_iter=100, random_state=1)
#lr = SGDClassifier(loss='log', max_iter=1000, random_state=1) # alternative scikit implementation with stochastic gradient (optimize cost function based on individual samples), loss='log_loss' after scikit-learn v1.1
lr.fit(X_train_std, y_train)


# predict and predict_proba tests
iP.print('Predict and predict_proba tests:')
iP.print(lr.predict_proba(X_test_std[:4, :])) # (5) probability of being in any (3) class for first 4 samples
iP.print(lr.predict_proba(X_test_std[:4, :]).sum(axis=1)) # (6) probabilities for all classes in sample should sum up to 1
iP.print(lr.predict_proba(X_test_std[:4, :]).argmax(axis=1)) # (7) get indexes of most-prob classes for first 4 samples
iP.print(lr.predict(X_test_std[:4, :])) # (8) same as above
iP.print(lr.predict(X_test_std[3, :].reshape(1, -1))) # (9) predict specified sample -> lr.predict(X_test_std[0, :]) without reshaping would throw ValueError (expected 2D array)


# plot results
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150), title='Scikit log-reg results')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('scikit log-reg.png', dpi=300)
plt.show()


# plot weights for various C=1/lambda (lambda -> strength of L2 regularization to tackle overfitting)
weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.figure(num='Weights for various L2 regularization strength')
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
#plt.savefig('scikit log-reg (regularization).png', dpi=300)
plt.show()