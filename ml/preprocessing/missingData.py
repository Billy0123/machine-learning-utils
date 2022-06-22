import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer

from ml.utils.indexPrinter import indexPrinter
iP = indexPrinter()


# import sample data
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))  # StringIO for stream/simulation purpose
iP.print(df) # (1) data frame with NaNs


# remove rows/columns with missing values
iP.print(df.isnull().sum()) # (2) count NaNs in columns
iP.print(df.values) # (3) access the underlying NumPy array via the `values` attribute
iP.print(df.dropna(axis=0)) # (4) remove rows that contain missing values
iP.print(df.dropna(axis=1)) # (5) remove columns that contain missing values
iP.print(df.dropna(how='all')) # (6) only drop rows where all columns are NaN
iP.print(df.dropna(thresh=4)) # (7) drop rows that have less than 4 real values
iP.print(df.dropna(subset=['C'])) # (8) only drop rows where NaN appear in specific columns (here: 'C')


# imputing missing values
iP.print(df.values) # (9) original array


# impute missing values via the column mean
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data_mean = imr.transform(df.values)
iP.print(imputed_data_mean) # (10) means imputed


# impute missing values via the column mean
imr = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_data_median = imr.fit_transform(df.values)
iP.print(imputed_data_median) # (11)  medians imputed


# impute missing values via the column mean
imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputed_data_most_frequent = imr.fit_transform(df.values)
iP.print(imputed_data_most_frequent) # (12) most_frequents imputed


# impute missing values via the column mean
imr = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=12345)
imputed_data_constant = imr.fit_transform(df.values)
iP.print(imputed_data_constant) # (13) constants imputed