import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# import sample data
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
iP.print(('Class labels', np.unique(df_wine['Class label']))) # (1) wine data set contains 3 different types (classes) of wine
iP.print(df_wine.head()) # (2) print first (auto=5) rows of data set
iP.print(df_wine.tail(8)) # (3) print last 8 rows of data set


# partitioning a dataset into a separate training and test set
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                     test_size=0.3, #30% test, 70% training (common: 60/40, 70/30, 80/20, but when big data set, even 90:10 or 99:1)
                     random_state=0,
                     stratify=y) #stratify: get the same proportions (approx) of various classes (y) in train/test sets
iP.print(('Labels counts in y:', np.bincount(y))) # (4)
iP.print(('Labels counts in y_train:', np.bincount(y_train))) # (5)
iP.print(('Labels counts in y_test:', np.bincount(y_test))) # (6)


# bringing features onto the same scale - normalization (in the sense of scaling to range: 0-1)
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test) # transforming with 'mms' fitted to the train data!


# bringing features onto the same scale - standardization (returned feature: <avg. value> = 0, <standard deviation> = 1)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test) # transforming with 'stdsc' fitted to the train data!


# a visual example:
ex = np.array([0, 1, 2, 3, 4, 5])
iP.print(('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))) # (7) normalized
iP.print(('standardized:', (ex - ex.mean()) / ex.std())) # (8) standarized