import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from ml.utils.regressionPlots import residualPlot
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# load housing data
df = pd.read_csv('../databases/housing.data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
iP.print(df.head()) # (1) show first rows of housing data


# Evaluating the performance of MULTI-linear regression models (target variable: MEDV, explanatory variables: all others)
X = df.iloc[:, :-1].values
y = df['MEDV'].values

# split data for train/test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# fit linear regression
lr = LinearRegression().fit(X_train, y_train)

# generate points and plot residuals (chould be pure-chaotic)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
residualPlot(y_train, y_train_pred, y_test, y_test_pred, title='Residual plot - default multi-linear regression')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()

# show MSE and R^2 scores
iP.print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred))) # (2) MeanSquaredError = Sum_i[(y_i_pred - y_i)^2] / n
iP.print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred))) # (3) R^2 = 1-SumSquaredErrors/SumSquaresTotal = 1 - (Sum_i[(y_i_pred - y_i)^2]) / (Sum_i[(y_i_pred - mu_y)^2]) = 1-MSE/Var(y)