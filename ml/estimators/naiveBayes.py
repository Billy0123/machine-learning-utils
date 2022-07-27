import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from manual_mixedNaiveBayes import MixedNaiveBayes

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# load sample data (credit approval data set)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data',
                 header=None)
# if the dataset is temporarily unavailable from the UCI machine learning repository, un-comment the following line of code to load the dataset from a local path:
#df = pd.read_csv('../databases/crx.data', header=None)
# also, various datasets are available at: https://archive.ics.uci.edu/ml/datasets.php


# preprocessing 1 - drop rows with missing data
pd.set_option('display.max_rows', None)  # let pandas display all rows
iP.print(df)  # (1) show the dataset  -> find that the missing values appear as '?'
iP.print(f"Before missing-val rows are dropped, len = {len(df)}")  # (2)
df = df[df != '?'].dropna(axis=0)  # drop rows containing nulls/NaNs (if condition df!='?' is false, cell-value will be NaN)
iP.print(f"After missing-val rows are dropped, len = {len(df)}")  # (3)

# preprocessing 2 - encode features for specific NB classifiers and create train/test subsets
iP.print(df.nunique())  # (4) count unique values for each feature (2 (binary) for Bernoulli, more for categorical, integers for complement (improved multinomial), reals for gaussian
iP.print(df.head())  # (5) all features before encoding some of them

IDsToEcode = [3,4,5,6,12,0,8,9,11]
oe = OrdinalEncoder().fit(df.loc[:, IDsToEcode].values)
X = df.loc[:, :14].values
X[:, IDsToEcode] = oe.transform(X[:, IDsToEcode])
iP.print(X[:5])  # (6) all features after encoding some of them

y = df.loc[:, 15].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)

# preprocessing 3 - create subsets of features for various types of Naive Bayes classifiers
X_train_NB = []
X_test_NB = []
IDsTypesAndClfs = np.array([
    [[1,2,7,13,14], float, GaussianNB()],
    [[3,4,5,6,12], int, CategoricalNB()],  # operate on LENGTH of values of features
    [[10], int, MultinomialNB()],  # operate on SUM of values of features
    [[10], int, ComplementNB()],  # operate on SUM of values of features (complement improves default multinomial classifier for imbalanced data sets)
    [[0,8,9,11], int, BernoulliNB()]],  # like categorical, but binary
    dtype=object)
for IDs, type_ in IDsTypesAndClfs[:, :2]:
    X_train_NB.append(np.array(X_train[:, IDs], dtype=type_))
    X_test_NB.append(np.array(X_test[:, IDs], dtype=type_))


# create, fit and check score of all Naive Bayes classifiers
clfs_NB = [IDsTypesAndClfs[i,2].fit(X_train_NB[i],y_train) for i in range(len(IDsTypesAndClfs))]
for i in range(len(clfs_NB)):
    print(f"({clfs_NB[i]}) "
          f"train accuracy: {clfs_NB[i].score(X_train_NB[i], y_train)}, "
          f"test accuracy: {clfs_NB[i].score(X_test_NB[i], y_test)}")

# mixed NB classifier using all NB defined classifiers
mnb = MixedNaiveBayes(*clfs_NB)
iP.print(f"({mnb}) "
          f"train accuracy: {mnb.score(X_train_NB, y_train)}, "
          f"test accuracy: {mnb.score(X_test_NB, y_test)}")  # (7)

# mixed NB classifier using only Categorical and Bernoulli classifiers
mnb = MixedNaiveBayes(clfs_NB[1],clfs_NB[4])
iP.print(f"({mnb}) "
          f"train accuracy: {mnb.score([X_train_NB[1],X_train_NB[4]], y_train)}, "
          f"test accuracy: {mnb.score([X_test_NB[1],X_test_NB[4]], y_test)}")  # (8)

# mixed NB classifier using only Bernoulli classifier
mnb = MixedNaiveBayes(clfs_NB[4])
iP.print(f"({mnb}) "
          f"train accuracy: {mnb.score([X_train_NB[4]], y_train)}, "
          f"test accuracy: {mnb.score([X_test_NB[4]], y_test)}")  # (9)