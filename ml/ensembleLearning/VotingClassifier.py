from sklearn.ensemble import VotingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from itertools import product
from sklearn.model_selection import GridSearchCV

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# import iris dataset
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]  # only features [1,2] and only 2 classes (first 50 is class0, next 50 is class1, next 50 is class2, 150 in total)
le = LabelEncoder()
y = le.fit_transform(y)  # encode labels, starting with 0

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,  # split 50/50
                                                    random_state=1,
                                                    stratify=y)

# create classifiers
clf1 = LogisticRegression(penalty='l2',
                          C=0.001,
                          random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')

# pipelines for logReg and KNN, but DecisionTree does not need standardization
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])
clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

# check mean C-V scoring of individual classifiers
iP.print('10-fold cross validation (individual):') # (1) individual scoring
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# create VotingClassifier (using scikit-learn)
mv_clf = VotingClassifier(estimators=[('lr',pipe1), ('dt',clf2), ('knn',pipe3)],
                          voting='soft')  # hard - majority voting, soft - based on the argmax of the sums of the predicted probabilities (use with well-calibrated classifiers)
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

iP.print('10-fold cross validation (individual+ensemble) on TRAIN set:') # (2) individual+ensemble scoring (train set)
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')  # roc_auc works only with 'soft' voting (not with hard - then use e.g. accuracy)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

iP.print('10-fold cross validation (individual+ensemble) on TEST set:') # (3) individual+ensemble scoring (test set)
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_test,
                             y=y_test,
                             cv=10,
                             scoring='roc_auc')  # roc_auc works only with 'soft' voting (not with hard - then use e.g. accuracy)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# Plot ROC_AUC graph for all classifiers
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
plt.figure(num='Receiver Operating Characteristic (ROC) [TPR vs FPR] and Area Under Curve (AUC)')
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1 (for purposes of roc_auc curve, which has to assume true-positives and false-positives rates
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()


# Plot regions of decisions for all classifiers
# at first, standardize train data -> pipelines of 'lr' and 'knn' do that, but 'dt' does not needed that -> however, for presentation purposes it is necessary also for 'dt'
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(7, 5),
                        num='Decision regions for various classifiers')

for idx, clf, tt in zip(product([0, 1], [0, 1]),all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
                                  X_train_std[y_train == 0, 1],
                                  c='blue',
                                  marker='^',
                                  s=50)

    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                  X_train_std[y_train == 1, 1],
                                  c='green',
                                  marker='o',
                                  s=50)
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -5.,
         s='Sepal width [standardized]',
         ha='center', va='center', fontsize=12)
plt.text(-12.5, 4.5,
         s='Petal length [standardized]',
         ha='center', va='center',
         fontsize=12, rotation=90)
plt.show()


# before we tune the individual classifier's parameters for ensemble classification, call get_params to get a basic idea of how we can access the individual parameters
iP.print(mv_clf.get_params()) # (4)

# params for GridSearch
params = {'dt__max_depth': [1, 2],
          'lr__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf,  # GridSearch for classifiers INSIDE ensemble classifier
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(X_train, y_train)

iP.print("Mean test scores of ensemble classifier with various sets of params of classifiers INSIDE it (params for 'dt' and 'lr' were tested)") # (5)
for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_['mean_test_score'][r],
             grid.cv_results_['std_test_score'][r] / 2.0,
             grid.cv_results_['params'][r]))

iP.print('Best parameters: %s' % grid.best_params_) # (6)
iP.print('Accuracy: %.2f' % grid.best_score_) # (7)

mv_clf = grid.best_estimator_  # or: mv_clf.set_params(**grid.best_estimator_.get_params())   ; '**' is for dictionary-like list of (named) parameters
# mv_clf.fit(X_train, y_train)
# Note that the line above is not necessary, because the best_estimator_ will already be refit to the complete training set because of the refit=True setting in GridSearchCV (refit=True by default).
iP.print('Test accuracy: %.3f' % mv_clf.score(X_test, y_test)) # (8) accuracy of the best estimator on TEST (not validation) set

iP.print(mv_clf) # (9)

'''
The majority vote approach implemented in this notebook is not to
be confused with stacking. The stacking algorithm can be understood
as a two-layer ensemble, where the first layer consists of individual
classifiers that feed their predictions to the second level, where another
classifier (typically logistic regression) is fit to the level-1 classifier
predictions to make the final predictions.
See: sklearn.ensemble.StackingClassifier (usage is very similar to VotingClassifier, but has also 'final_estimator' parameter (default: log-reg))
'''