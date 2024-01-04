import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from imblearn.over_sampling import SMOTE

#checking the data
#val_x = pd.read_excel(open('p-1.xlsx', 'rb')) 
val_x = pd.read_csv('P1.csv')
val = pd.DataFrame()
val['SEn'] = val_x.SEn
val['CLASS'] = val_x.CLASS
X_1 = val[val['CLASS'] == 1] 
X_2 = val[val['CLASS'] == 2]
X_3 = val[val['CLASS'] == 3]
from sklearn.svm import OneClassSVM
clf = OneClassSVM(gamma='auto').fit(X_1)
k= clf.predict(X_1)
n_error_train = X_1[k == -1]
good_train = X_1[k != -1]
val['id'] = val_x.id
val['bis'] = val_x.bis
val['patient_id'] = val_x.patient_id
X_1['id'] = val['id'][val['CLASS'] == 1]
X_1['bis'] = val['bis'][val['CLASS'] == 1]
X_1['patient_id'] = val['patient_id'][val['CLASS'] == 1]
good_train = X_1[k != -1]
q1 = good_train

from sklearn.svm import OneClassSVM
clf = OneClassSVM(gamma='auto').fit(X_2)
k= clf.predict(X_2)
n_error_train = X_2[k == -1]
good_train = X_2[k != -1]
X_2['id'] = val['id'][val['CLASS'] == 2]
X_2['bis'] = val['bis'][val['CLASS'] == 2]
X_2['patient_id'] = val['patient_id'][val['CLASS'] == 2]
good_train = X_2[k != -1]
q2 = good_train

from sklearn.svm import OneClassSVM
clf = OneClassSVM(gamma='auto').fit(X_3)
k= clf.predict(X_3)
n_error_train = X_3[k == -1]
good_train = X_3[k != -1]
X_3['id'] = val['id'][val['CLASS'] == 3]
X_3['bis'] = val['bis'][val['CLASS'] == 3]
X_3['patient_id'] = val['patient_id'][val['CLASS'] == 3]
good_train = X_3[k != -1]
q3 = good_train
concat = pd.concat([q1,q2,q3],axis=0)

X_train = concat.iloc[:,[0,2]]
y_train = concat.iloc[:,[1]].values
X_res, Y_res = SMOTE(random_state = 42).fit_resample(X_train, y_train)
X_res['CLASS'] = Y_res
concat.to_csv('p-1_svm.csv')
X_res.to_csv('p1smote.csv')

