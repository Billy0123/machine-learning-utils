import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from ml.utils.plotDecisionRegions import plot_decision_regions
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


# Supervised data compression via scikit's linear discriminant analysis -> manual LDA is very similar to PCA (creation of transformation W matrix, but using mean vectors of individual classes [supervised!])
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

# Training logistic regression classifier using the first n_components reduced features.
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

# plot classification results of log-reg using reduced features set for train data set
plot_decision_regions(X_train_lda, y_train, classifier=lr, title='Log-reg classification using reduced (scikit LDA) features set - train data set')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('fe_scikit_LDA log-reg classification results train data set.png', dpi=300)
plt.show()

# plot classification results of log-reg using reduced features set for test data set
plot_decision_regions(X_test_lda, y_test, classifier=lr, title='Log-reg classification using reduced (scikit LDA) features set - test data set')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('fe_scikit_LDA log-reg classification results test data set.png', dpi=300)
plt.show()