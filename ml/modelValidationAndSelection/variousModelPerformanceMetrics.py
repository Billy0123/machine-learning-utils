import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from sklearn.utils import resample

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

# Create pipeline
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))


# Looking at different performance evaluation metrics
# Reading a confusion matrix (TruePositive,TrueNegative [diagonal], FalsePositive,FalseNegative [nondiagonal])
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
iP.print(confmat) # (4) confusion matrix

fig, ax = plt.subplots(figsize=(2.5, 2.5),num='Confusion matrix')
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()


# Additional Note: remember that we previously encoded the class labels so that:
# *malignant* samples are the "postive" class (1), and
# *benign* samples are the "negative" class (0):
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
iP.print(confmat) # (5)

# Note that the (true) class 0 samples that are correctly predicted as class 0 (true negatives) are now in the upper left corner of the matrix (index 0, 0). In order to change the ordering so that the true negatives are in the lower right corner (index 1,1) and the true positves are in the upper left, we can use the `labels` argument like shown below:
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0])
iP.print(confmat) # (6)
# Conclude: assuming that class 1 (malignant) is the positive class in this example, our model correctly classified 71 of the samples that belong to class 0 (true negatives) and 40 samples that belong to class 1 (true positives), respectively. However, our model also incorrectly misclassified 1 sample from class 0 as class 1 (false positive), and it predicted that 2 samples are benign although it is a malignant tumor (false negatives).


# Optimizing the precision and recall of a classification model -> F1 score, via GridSearch
# error: ERR = (FP+FN)/(FP+FN+TP+TN)
# accuracy: ACC = (TP+TN)/(FP+FN+TP+TN) = 1-ERR
# true-positive rate: TPR = TP/P = TP/(TP+FN)
# false-positive rate: FPR = FP/N = FP/(FP+TN)
iP.print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred)) # (7) PRE = TP/(TP+FP)
iP.print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred)) # (8) REC = TPR = TP/(TP+FN)
iP.print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred)) # (9) F1 = 2*PRE*REC/(PRE+REC)

scorer = make_scorer(f1_score, pos_label=0) # make user-defined scorer with specified class-label taken as positive
c_gamma_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
iP.print(gs.best_score_) # (10)
iP.print(gs.best_params_) # (11)


# Plotting a receiver operating characteristic and area under curve (ROC and AUC, 50% - random guess)
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2',
                                           random_state=1,
                                           C=100.0))
X_train2 = X_train[:, [4, 14]] # using only 2 features from original dataset to make ROC-AUC characteristics more interesting
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train)) # will plot 3 various folds (train/validate sets)

fig = plt.figure(figsize=(7, 5),num='Receiver Operating Characteristic (ROC) [TPR vs FPR] and Area Under Curve (AUC)')
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros(mean_fpr.shape[0])
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp1d(fpr, tpr, kind='linear')(mean_fpr)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= len(cv)
mean_tpr[0] = 0.0
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# The scoring metrics for multiclass classification: by default for multiclass, macro-average is used:
'''The weighted macro-average is calculated by weighting the score of
each class label by the number of true instances when calculating the average. The
weighted macro-average is useful if we are dealing with class imbalances, that is,
different numbers of instances for each label.
Micro-averaging is useful if we want to weight each instance or prediction equally.
'''
pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1,
                         greater_is_better=True,
                         average='micro') # for micro-averaging


# Dealing with class imbalance
X_imb = np.vstack((X[y == 0], X[y == 1][:40])) # imbalanced X: all class-0 and only first 40 class-1 members
y_imb = np.hstack((y[y == 0], y[y == 1][:40])) # imbalanced y: all class-0 and only first 40 class-1 members

y_pred = np.zeros(y_imb.shape[0])
iP.print(np.mean(y_pred == y_imb)) # (12) dummy prediction -> always 0-class (i.e. dominant) -> this is why ACCURACY scoring can be bad with imbalanced sets

# Resampling -> upsampling (randomly generate additional non-dominant-class samples up to the number of dominant-class samples)
iP.print(('Number of class 1 samples before:', X_imb[y_imb == 1].shape[0]))# (13)
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)
iP.print(('Number of class 1 samples after:', X_upsampled.shape[0])) # (14)

# create balanced samples using all-original class-0 samples and upsampled sets of class-1
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

y_pred = np.zeros(y_bal.shape[0])
iP.print(np.mean(y_pred == y_bal)) # (15) dummy prediction no longer works (50% accuracy)

# Resampling -> downsampling (randomly drop dominant-class samples down to the number of non-dominant-class samples)
iP.print(('Number of class 0 samples before:', X_imb[y_imb == 0].shape[0])) # (16)
X_downsampled, y_downsampled = resample(X_imb[y_imb == 0],
                                    y_imb[y_imb == 0],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 1].shape[0],
                                    random_state=123)
iP.print(('Number of class 0 samples after:', X_downsampled.shape[0])) # (17)

# create balanced samples using prev-imbalanced set of class-1 and downsampled sets of class-0
X_bal = np.vstack((X_downsampled, X_imb[y_imb==1]))
y_bal = np. hstack((y_downsampled, y_imb[y_imb==1]))

y_pred = np.zeros(y_bal.shape[0])
iP.print(np.mean(y_pred == y_bal)) # (15) dummy prediction no longer works (50% accuracy)