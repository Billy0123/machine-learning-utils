import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# create some non-linear scatter points
X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis] # generate two-dimensional array with np.newaxis to satisfy sklearn-like fit() methods
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])

# fit linearly
X_lin= np.arange(250, 600, 10)[:, np.newaxis]
lr = LinearRegression().fit(X, y)
y_lin_fit = lr.predict(X_lin)

# fit polynomially (turning linear regression into polynomial)
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X) # adjust lin-X values for quadratic fitting
pr = LinearRegression().fit(X_quad, y) # fit LINEARLY data, but using quadratic X set
y_quad_fit = pr.predict(quadratic.fit_transform(X_lin)) # predict using in-fly-transformed X_lin set

# plot results
plt.figure(num='Comparison of linear and quadratic fits')
plt.scatter(X, y, label='training points')
plt.plot(X_lin, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_lin, y_quad_fit, label='quadratic fit') # note that for plotting purposes X_lin is used (X_quad only for fitting!)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# check performances
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
iP.print('Training MSE linear: %.3f, quadratic: %.3f' % (
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred))) # (1) MES (see evaluatingPerformance notebook)
iP.print('Training R^2 linear: %.3f, quadratic: %.3f' % (
        r2_score(y, y_lin_pred),
        r2_score(y, y_quad_pred))) # (2) R^2 (see evaluatingPerformance notebook)


# load housing data
df = pd.read_csv('../databases/housing.data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
iP.print(df.head()) # (3) show first rows of housing data

# take only 'MEDV' and 'LSTAT' features, which are non-linearly-dependent (see pairRelationshipsOfFeatures notebook)
X = df[['LSTAT']].values # two-dimensional array to satisfy sklearn-like fit() methods
y = df['MEDV'].values


# fit features polynomially with orders=1,2,3
X_lin = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = LinearRegression().fit(X, y)
y_lin_fit = regr.predict(X_lin)
linear_r2 = r2_score(y, regr.predict(X))

quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)
regr = LinearRegression().fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_lin))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

cubic = PolynomialFeatures(degree=3)
X_cubic = cubic.fit_transform(X)
regr = LinearRegression().fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_lin))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# plot results
plt.figure(num='Comparison of various polynomial fits on MEDV = f(LSTAT)')
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_lin, y_lin_fit,
         label='linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue',
         lw=2,
         linestyle=':')
plt.plot(X_lin, y_quad_fit,
         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red',
         lw=2,
         linestyle='-')
plt.plot(X_lin, y_cubic_fit,
         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green',
         lw=2,
         linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')
plt.show()


# fit features custom-non-linearly, here: assume that MEDV=f(LSTAT) is approximated well by f(x)=2^-x -> nat-log of exp function is straight line, so: log(f(x))=-x
X_log = np.log(X)
y_sqrt = np.sqrt(y) # it's only intuition that sqrt on Y will fit well

# fit features
X_lin = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

regr = LinearRegression().fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_lin)
exp_r2 = r2_score(y_sqrt, regr.predict(X_log))

# plot results
plt.figure(num='Custom-non-linear (here, exponential) fit')
plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
plt.plot(X_lin, y_lin_fit,
         label='exponential (linear when used with log on X scale (d=1)), $R^2=%.2f$' % exp_r2,
         color='blue',
         lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()