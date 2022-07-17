import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


# Fine-tuning machine learning models via grid search - tuning hyperparameters via grid search (checks all the combinations of parameters and returns best estimator <- only validation sets, NOT TESTS)
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1, # all the available cores
                  refit=True)
gs = gs.fit(X_train, y_train)
iP.print(gs.best_score_) # (4) best score
iP.print(gs.best_params_) # (5) best parameters

clf = gs.best_estimator_
# clf.fit(X_train, y_train)
# Note that the line above is not necessary, because the best_estimator_ will already be refit to the complete training set because of the refit=True setting in GridSearchCV (refit=True by default).
iP.print('Test accuracy: %.3f' % clf.score(X_test, y_test)) # (6) accuracy of the best estimator on TEST (not validation) set

'''GridSearchCV is very powerful but also - computationally - very expensive. Alternative approach: 
from sklearn.model_selection import RandomizedSearchCV
'''


# Algorithm selection with nested cross-validation - the 'outer' CV (5) generate train/test sets from 'original train tests' while the 'inner' CV (2) generate train/validate sets from single-train set
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
iP.print('(SVM) CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores))) # (7) accuracy
gs = gs.fit(X_train, y_train)
iP.print(gs.best_score_) # (8)
iP.print(gs.best_params_) # (9)

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
iP.print('(DecisionTree) CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores))) # (10)
gs = gs.fit(X_train, y_train)
iP.print(gs.best_score_) # (11)
iP.print(gs.best_params_) # (12)