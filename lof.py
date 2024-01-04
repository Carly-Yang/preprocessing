# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
# Importing the dataset
dataset = pd.read_csv('train.csv')
all = pd.DataFrame()
eeg = dataset.eeg
bis = dataset.bis
all["bis"] = bis
all["eeg"] = eeg
# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(all)
# select all rows that are not outliers

clf = LocalOutlierFactor()
b = clf.fit_predict(all)

mask_all = all[yhat != -1] 
delete_number = all[yhat == -1] 
outlier =clf.negative_outlier_factor_