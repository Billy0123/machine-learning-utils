import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from manual_sequentialBackwardSelection import SBS
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# import and standardize the wine data
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
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                     test_size=0.3, #30% test, 70% training (common: 60/40, 70/30, 80/20, but when big data set, even 90:10 or 99:1)
                     random_state=0,
                     stratify=y) #stratify: get the same proportions (approx) of various classes (y) in train/test sets
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test) # transforming with 'stdsc' fitted to the train data!


# L1-regularization
lr = LogisticRegression(penalty='l1', C=1.0, solver='saga') # L2-regularization (circle-like) is 'softer' than L1 (diamond-like), which faster generate sparse data (weights=0); default solver (lbfgs does not support L1!)
# C=1.0 is the default. Increase or decrease it to make the regularization effect stronger or weaker, respectively.
lr.fit(X_train_std, y_train)
iP.print(('Training accuracy:', lr.score(X_train_std, y_train))) # (1)
iP.print(('Test accuracy:', lr.score(X_test_std, y_test))) # (2)
iP.print(lr.intercept_) # (3) coefs w[0] (bias/offset) in the dot product z=w[0]*1+w[1]*x[1]+...+w[m]*x[m]; 3 intercepts, because of OneVsRest strategy (3 classes not 2)
np.set_printoptions(8)
iP.print(lr.coef_.shape) # (4) 3(classes)x13(features)
iP.print(lr.coef_) # (5) L1-regularized weights (sparse arrays)

# plot weights vs C by L1-regularization
fig = plt.figure(num='Weights vs C(=1/lambda) by L1-regularization')
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0, solver='saga')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
# plt.savefig('weights_vs_C_L1-regularization.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()


# sequential feature selection algorithm - Sequential Backward Selection (manual implementation) on KNN example
knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

fig = plt.figure(num='Accuracy of KNN vs number of features by SBS')
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('accuracy_vs_numberOfFeats_SBS.png', dpi=300)
plt.show()

#KNN weights plot analysis (SBS)
k3 = list(sbs.subsets_[-3])
iP.print(df_wine.columns[1:][k3]) # (6) which features were chosen at k_features=3?

knn.fit(X_train_std, y_train)
iP.print(('Training accuracy (all 13 features):', knn.score(X_train_std, y_train))) # (7) score on train data for all (13) features
iP.print(('Test accuracy (all 13 features):', knn.score(X_test_std, y_test))) # (8) score on test data for all (13) features

knn.fit(X_train_std[:, k3], y_train)
iP.print(('Training accuracy (3 selected features):', knn.score(X_train_std[:, k3], y_train))) # (9) score on train data for selected (3) features
iP.print(('Test accuracy (3 selected features):', knn.score(X_test_std[:, k3], y_test))) # (10) score on test data for selected (3) features


# assessing feature importance with Random Forests
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# plot feature importances
iP.index+=1
print("%d:" % iP.index) # (11) list importances of all features in sorted (descending) order
for f in range(len(indices)):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
print("-----")

fig = plt.figure(num='Importances of features by Random Forest')
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
# plt.savefig('feat_importances_RandomForest.png', dpi=300)
plt.show()

# select features with importances >=0.1
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)

iP.index += 1
print("%d:" % iP.index) # (12) list importances that meet the criterion (>=0.1)
print(('Number of features that meet this threshold criterion:', X_selected.shape[1]))
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
print("-----")