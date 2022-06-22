import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# import sample data
df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
iP.print(df) # (1) print data frame


# mapping ordinal features
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)
iP.print(df) # (2) print data frame with mapped sizes

# invert size mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
iP.print(df) # (3) print data frame with reverted mapping of sizes


# encoding class labels
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))} # create a mapping dict to convert class labels from strings to integers
iP.print(class_mapping) # (4) print mapping dict
df['classlabel'] = df['classlabel'].map(class_mapping)
iP.print(df) # (5) print data frame with encoded class labels

# reverse the class label mapping
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
iP.print(df) # (6) print data frame with reverted encoding of class labels


# Label encoding with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
iP.print(y) # (7) print encoding array; to change in data frame: df['classlabel'] = class_le.fit_transform(df['classlabel'].values)

# reverse mapping via sklearn's LabelEncoder
iP.print(class_le.inverse_transform(y)) # (8) print reverse-encoding array


# one-hot encoding on nominal features
X = df[['color', 'size', 'price']].values
iP.print(X) # (9) show array color/size/price
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
iP.print(X) # (10) BAD because ML algorithms will treat e.g. red>green>blue -> solution is one-hot encoding (column for each color & binary values)

X = df[['color', 'size', 'price']].values # re-initiate X
ohe = OneHotEncoder()
X_hot_encoded = ohe.fit_transform(X[:,0].reshape(-1,1)).toarray() # select [0] column (color) for one-hot encoding; reshape, because fit_transform's argument has to be 2D-array in this case; order of resulting binary-table refers to alphabetic order (blue|green|red)
X = np.concatenate((X_hot_encoded,X[:,1:]),axis=1)
iP.print(X) # (11) ndarray
iP.print(pd.DataFrame(X,columns=ohe.categories_[0].tolist()+['size','price'])) # (12) pandas data frame

# return dense array to skip the <toarray> step
X = df[['color', 'size', 'price']].values # re-initiate X
ohe = OneHotEncoder(sparse=False)
X_hot_encoded = ohe.fit_transform(X[:,0].reshape(-1,1)) # select [0] column (color) for one-hot encoding; reshape, because fit_transform's argument has to be 2D-array in this case; order of resulting binary-table refers to alphabetic order (blue|green|red)
X = np.concatenate((X_hot_encoded,X[:,1:]),axis=1)
iP.print(X) # (13) ndarray
iP.print(pd.DataFrame(X,columns=ohe.categories_[0].tolist()+['size','price'])) # (14) pandas data frame


# one-hot encoding via pandas
iP.print(pd.get_dummies(df[['price', 'color', 'size']])) # (15) pandas read 'size' as categorical value and automatically one-hot encodes it
df['size'] = df['size'].map(size_mapping)
iP.print(pd.get_dummies(df[['price', 'color', 'size']])) # (16) when size is properly mapped, only 'size' is one-hot encoded


# multicollinearity guard for the OneHotEncoder
X = df[['color', 'size', 'price']].values # re-initiate X
ohe = OneHotEncoder(sparse=False, drop='first')
X_hot_encoded = ohe.fit_transform(X[:,0].reshape(-1,1)) # select [0] column (color) for one-hot encoding; reshape, because fit_transform's argument has to be 2D-array in this case; order of resulting binary-table refers to alphabetic order (blue|green|red)
X = np.concatenate((X_hot_encoded,X[:,1:]),axis=1)
iP.print(pd.DataFrame(X,columns=ohe.categories_[0].tolist()[1:]+['size','price'])) # (17) drop single collinear binary feature (no information lost)

# multicollinearity guard in get_dummies
iP.print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)) # (18) drop single collinear binary feature (no information lost)