import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# Bagging - Building an ensemble of classifiers from bootstrap samples
'''
Bagging can deal with overfitting (separation between train and test scores).
Each classifier receives a random subset (with replacement)
of samples from the training set. Each subset contains a certain portion of duplicates
and some of the original samples don't appear in a resampled dataset at all due to
sampling with replacement. Once the individual classifiers are fit to the bootstrap
samples, the predictions are combined using majority voting.
(!) Note that bagging is also related to the random forest classifier. In fact, random
forests are a special case of bagging where we ALSO use random feature subsets when
fitting the individual decision trees (here, pure-bagging: all features are used.
'''


# Applying bagging to classify samples in the Wine dataset
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

# if the Wine dataset is temporarily unavailable from the UCI machine learning repository, un-comment the following line of code to load the dataset from a local path:
#df_wine = pd.read_csv('../databases/wine.data', header=None)
# also, various datasets are available at: https://archive.ics.uci.edu/ml/datasets.php

# drop 1 class
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

# encode labels starting from 0
le = LabelEncoder()
iP.print(y) # (1) before
y = le.fit_transform(y)
iP.print(y) # (2) after

# split train/test data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)


# create very deep decisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=None,
                              random_state=1)

# create bagging classifier consisting of 500 deep trees using random subsets
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)


# check score of single, deep-tree
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
tree_train = accuracy_score(y_train, y_train_pred)
y_test_pred = tree.predict(X_test)
tree_test = accuracy_score(y_test, y_test_pred)
iP.print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test)) # (3) single deep-tree scores

# check score of bagged trees
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
bag_train = accuracy_score(y_train, y_train_pred)
y_test_pred = bag.predict(X_test)
bag_test = accuracy_score(y_test, y_test_pred)
iP.print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test)) # (4) bagged trees scores


# draw decision regions for single deep-tree and bagged trees
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3),
                        num='Comparison of decision regions for single deep-tree and bagged trees')

for idx, clf, tt in zip([0, 1],
                        [tree, bag],
                        ['Decision tree', 'Bagging']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train == 0, 0],
                       X_train[y_train == 0, 1],
                       c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train == 1, 0],
                       X_train[y_train == 1, 1],
                       c='green', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -0.5,
         s='OD280/OD315 of diluted wines',
         ha='center', va='center', fontsize=12)
plt.tight_layout()
plt.show()