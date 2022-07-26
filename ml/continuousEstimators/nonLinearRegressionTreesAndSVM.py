import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.utils.regressionPlots import regressionPlot,residualPlot
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# load housing data
df = pd.read_csv('../databases/housing.data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
iP.print(df.head()) # (1) show first rows of housing data

# take only 'MEDV' and 'LSTAT' features, which are non-linearly-dependent (see pairRelationshipsOfFeatures notebook)
X = df[['LSTAT']].values # two-dimensional array to satisfy sklearn-like fit() methods
y = df['MEDV'].values


# Decision tree regression
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

# plot results
sort_idx = X.flatten().argsort()
regressionPlot(X[sort_idx], y[sort_idx], [tree], title='Decision Tree non-linear fit')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()


# take all features for random forest regression and SVM; split for train/test subsets - target variable: MEDV, explanatory variables: all others
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)


# Random forest regression
forest = RandomForestRegressor(n_estimators=1000,
                               criterion='squared_error',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

iP.print('(random forest) MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred))) # (2) MES (see evaluatingPerformance notebook)
iP.print('(random forest) R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred))) # (3) R^2 (see evaluatingPerformance notebook)

# plot residuals
residualPlot(y_train,y_train_pred,y_test,y_test_pred,title='Residual plot - random forest regresion')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()


# SVM regression
svmreg_pip = make_pipeline(StandardScaler(),
                           SVR(
                               C=100.0,
                               epsilon=0.2,
                               kernel='rbf'
                           ))
svmreg_pip.fit(X_train, y_train)
y_train_pred = svmreg_pip.predict(X_train)
y_test_pred = svmreg_pip.predict(X_test)

iP.print('(SVM) MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred))) # (4) MES (see evaluatingPerformance notebook)
iP.print('(SVM) R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred))) # (5) R^2 (see evaluatingPerformance notebook)

# plot residuals
residualPlot(y_train,y_train_pred,y_test,y_test_pred,title='Residual plot - SVM')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()