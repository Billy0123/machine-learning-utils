import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# load sample data
'''df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                 header=None)'''
# if the Breast Cancer dataset is temporarily unavailable from the UCI machine learning repository, un-comment the following line of code to load the dataset from a local path:
df = pd.read_csv('../databases/wdbc.data', header=None)
# also, various datasets are available at: https://archive.ics.uci.edu/ml/datasets.php

iP.print(df.head()) # (1) show head of dataset

# encode class labels
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
iP.print(le.classes_) # (2) check what classes are encoded
iP.print(le.transform(['M', 'B'])) # (3) test of encoding

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)


# Combining transformers and estimators in a pipeline
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1)) # args: (*transformers,estimator)
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
iP.print(y_pred) # (4) predictions
iP.print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test)) # (5) score


# K-fold cross-validation - stratified (keep proportions of classes in train/test sets)
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train) # 10 is typical (good) value -> less for big sets (computing time), larger for small sets (smaller validation sets, but larger train)
iP.print("K-fold cross validation:") # (6)
scores = []
for k, (train, validate) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[validate], y_train[validate])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k + 1,
                                                     np.bincount(y_train[train]), score))
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# Automated C-V score checker using cross_val_score
iP.print("Automated C-V score using cross_val_score:")
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))