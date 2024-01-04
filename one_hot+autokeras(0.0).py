# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 21:23:57 2022

@author: kai Yang
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from imblearn.over_sampling import SMOTE

#checking the data
dataset = pd.read_csv('train_nor.csv')
X = dataset.iloc[:, 0:4]
#feature_train = X.drop(['bp'], axis=1)
y = dataset.iloc[:, 4].values
X_test = X.iloc[0:382, :] 
X_train = X.iloc[383:2901, :] 
X_val = X.iloc[2901:3127, :] 
y_test = y[0:382]
y_train = y[383:2901]
y_val = y[2901:3127]

#y = np_utils.to_categorical(y)
y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)
X_res, Y_res = SMOTE(random_state = 42).fit_resample(X, y)
X_res['SCORE'] = Y_res
#X = sc.transform(X)
####### normal Classifier###########
# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout
model = Sequential([
    tf.keras.Input(shape=(4,)),
    Dense(64, activation='relu'),# 8個隐藏神经元的全连接层
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='sigmoid'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='sigmoid'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='sigmoid'),
    Dropout(0.5),
    Dense(5, activation='softmax')])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_res, Y_res, batch_size = 64, epochs = 20, validation_data =(X_val, y_val))
loss, accuracy = model.evaluate(X_val, y_val)
model.summary()

# ------ Plot loss -----------#
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# ------ Plot accuracy -----------#
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
pred = model.predict(X_test)
#pred = model.predict(X_val)
#X_pred = model.predict(X)
np.max(pred, axis=1)
#np.max(X_pred, axis=1)
y_pred_1 = tf.argmax(pred, axis=1)
#y_pred_1 = tf.argmax(X_pred, axis=1)
y_test = tf.argmax(y_test, axis=1)
#y_test = tf.argmax(y_val, axis=1)
y_test = np.asarray(y_test).astype('float32')
y_pred_1 = np.asarray(y_pred_1).astype('float32')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_1)
all_cm = sum(sum(cm))
acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/all_cm
acc

# save model
model.save('50_patient.h5')


####### autokeras Classifier###########
import autokeras as ak
clf = ak.StructuredDataClassifier(max_trials=300, overwrite=True)
clf.fit(x=X_res, y=Y_res,validation_data=(X_val, y_val), epochs=20)

# Evaluate the best model with testing data
loss, acc = clf.evaluate(X_val, y_val)

# Visualizing the best model
model = clf.export_model()
model.summary()

# Save the best model
model.save("ak_50patient-30.h5", save_format="tf")

# Reload the model to make predictions
from tensorflow.keras.models import load_model
loaded_model = load_model("ak_50patient-25.h5")
pred = loaded_model.predict(X_test)
y_pred_1 = tf.argmax(pred, axis=1)
#y_test = tf.argmax(y_test, axis=1)
y_test = np.asarray(y_test).astype('float32')
y_pred_1 = np.asarray(y_pred_1).astype('float32')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_1)
all_cm = sum(sum(cm))
acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/all_cm
acc