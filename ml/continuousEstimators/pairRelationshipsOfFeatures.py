import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# load housing data
df = pd.read_csv('../databases/housing.data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
iP.print(df.head()) # (1) show first rows of housing data


# plot pair-relationships of 5 chosen features from entire dataset
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], height=1.5)
plt.tight_layout()
plt.show()


# plot heatmap (correlation matrix - which is standardized covariance matrix (ranges from -1 to 1))
cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.tight_layout()
plt.show()