import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# Leveraging weak learners via adaptive boosting
'''
In contrast to bagging, the initial formulation of boosting, the algorithm uses random
subsets of training samples drawn from the training dataset without replacement;
the original boosting procedure is summarized in the following four key steps:
1. Draw a random subset of training samples d_1 without replacement from
training set D to train a weak learner C_1.
2. Draw a second random training subset d_2 without replacement from
the training set D and add 50 percent of the samples that were previously
misclassified (by C_1) to train a weak learner C_2.
3. Find the training samples d_3 in training set D, which C_1 and C_2 disagree
upon, to train a third weak learner C_3.
4. Combine the weak learners C_1, C_2, and C_3 via majority voting.
Boosting can lead to a decrease in bias as well as variance compared to bagging
models. In practice, however, boosting algorithms such as AdaBoost are also
known for their high variance, that is, the tendency to overfit the training data.
In contrast to the original boosting procedure as described above, AdaBoost uses
the complete training set D (not subsets d_i) to train the weak learners where the training samples are
reweighted in each iteration to build a strong classifier that learns from the mistakes
of the previous weak learners in the ensemble. 
'''


# Applying AdaBoost to classify samples in the Wine dataset
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


# create stump-tree classifier (weak learner)
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=1,
                              random_state=1)

# create adaBoost classifier consisting of 500 stump-trees
ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)


# check score of single, deep-tree
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
tree_train = accuracy_score(y_train, y_train_pred)
y_test_pred = tree.predict(X_test)
tree_test = accuracy_score(y_test, y_test_pred)
iP.print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test)) # (3) single deep-tree scores

# check score of bagged trees
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
ada_train = accuracy_score(y_train, y_train_pred)
y_test_pred = ada.predict(X_test)
ada_test = accuracy_score(y_test, y_test_pred)
iP.print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test)) # (4) adaboosted trees scores


# draw decision regions for single deep-tree and adaboosted trees
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3),
                        num='Comparison of decision regions for single deep-tree and adaboosted trees')

for idx, clf, tt in zip([0, 1],
                        [tree, ada],
                        ['Decision tree', 'AdaBoost']):
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