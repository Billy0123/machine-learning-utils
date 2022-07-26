import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from ml.utils.regressionPlots import regressionPlot
from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# load housing data
df = pd.read_csv('../databases/housing.data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
iP.print(df.head()) # (1) show first rows of housing data

# take only 'MEDV' and 'RM' features, which are (approximately) linearly-dependent (see pairRelationshipsOfFeatures notebook)
X = df[['RM']].values # two-dimensional array to satisfy sklearn-like fit() methods
y = df['MEDV'].values
# all of the regression estimators below can be used in multi-linear form (see: evaluatingPerformance notebook)


# compute linear regression using scikit-learn
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
iP.print('(lin-reg) Slope: %.3f' % lr.coef_[0]) # (2) lin-reg slope
iP.print('(lin-reg) Intercept: %.3f' % lr.intercept_) # (3) lin-reg intercept


# plot linear regression with scatter points
regressionPlot(X, y, [lr], title='MEDV = f(RM)')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()


# Fitting a robust regression model using RANSAC (alternative to throwing out outliers - RANdom SAmple Consensus which tries to fit only to data recognized as inliers)
'''
Summarized iterative RANSAC algorithm is as follows:
1. Select a random number of samples to be inliers and fit the model.
2. Test all other data points against the fitted model and add those points that fall within a user-given tolerance to the inliers.
3. Refit the model using all inliers.
4. Estimate the error of the fitted model versus the inliers.
5. Terminate the algorithm if the performance meets a certain user-defined threshold or if a fixed number of iterations were reached; go back to step 1 otherwise.
'''
ransac = RANSACRegressor(LinearRegression(), # you can choose different regression model
                         max_trials=100,
                         min_samples=50,
                         loss='absolute_error',
                         residual_threshold=5.0,
                         random_state=0)
ransac.fit(X, y)

# plot RANSAC regression results and show inliers (taken into account) and outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.figure(num='MEDV = f(RM); RANSAC regression')
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()

iP.print('(RANSAC\'s lin-reg) Slope: %.3f' % ransac.estimator_.coef_[0]) # (4) RANSAC's lin-reg slope
iP.print('(RANSAC\'s lin-reg) Intercept: %.3f' % ransac.estimator_.intercept_) # (5) RANSAC's lin-reg intercept


# Regularized methods for regression
'''
Ridge regression: L2 penalized model, cost function: J(w) = Sum_i[(y_i_pred-y_i)^2] + lambda*||w||^2_2
LeastAbsoluteShrinkageandSelectionOperator(LASSO): L1 penalized model (useful as a supervised feature selection technique), cost function: J(w) = Sum_i[(y_i_pred-y_i)^2] + lambda*||w||_1
ElasticNet: merge L2 and L1 penalties (overcome some limitations of LASSO, such as number of selected variables), cost function: J(w) = Sum_i[(y_i_pred-y_i)^2] + lambda_2*w^2_2 + lambda_1*w_1
'''
ridge = Ridge(alpha=1.0).fit(X, y) # Ridge regression
lasso = Lasso(alpha=1.0).fit(X, y) # LASSO regression
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X, y) # Elastic Net regression

# plot all the regression models with scatter points
regressionPlot(X, y, [lr,ransac,ridge,lasso,elanet], title='various regression models MEDV = f(RM)')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()

# plot slope coefficients of various models
iP.print((lr.coef_[0],
          ransac.estimator_.coef_[0],
          ridge.coef_[0],
          lasso.coef_[0],
          elanet.coef_[0])) # (6)