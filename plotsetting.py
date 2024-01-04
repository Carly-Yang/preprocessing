import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


val = pd.read_csv('val_result.csv')
train = pd.read_csv('result.csv')

########lof#########
from sklearn.neighbors import LocalOutlierFactor
all = pd.DataFrame()
eeg = train.eeg
bis = train.bis
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


###############standard - eeg
feature_train = mask_all.eeg
feature_val = val.eeg
scaler = StandardScaler()
stand_train = pd.DataFrame(mask_all, columns=['eeg'])
stand_val = pd.DataFrame(val, columns=['eeg'])

df_stand_train = pd.DataFrame(stand_train, columns=['eeg'])
df_stand_val = pd.DataFrame(stand_val, columns=['eeg'])

########bis
mask_all['bis_used'] = mask_all.bis / 100
val['bis_used'] = val['bis'] / 100

X_train = df_stand_train['eeg']
X_val = df_stand_val['eeg']
y_train = mask_all['bis_used']
y_val = val['bis_used']

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error

model = Sequential([
    tf.keras.Input(shape=(1,)),
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
    Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
history = model.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data =(X_val, y_val))
model.summary()
loss, acc = model.evaluate(X_val, y_val)
print('history dict:', history.history)

##########plot
n = np.arange(1, len(history.history['loss']) + 1)
plt.figure(figsize=(10, 6))
plt.plot(n, history.history['loss'], label='train_loss')
plt.plot(n, history.history['val_loss'], label='val_loss')
plt.title('Training and Validation Loss', fontsize=16)
plt.xlabel('Epoch #', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend()
plt.grid(True)

##########pred
dd = pd.read_csv('9_step3.csv')

feature_dd = dd.eeg
scaler = StandardScaler()
feature_dd = pd.DataFrame(feature_dd)
stand_dd = scaler.fit_transform(feature_dd)
df_stand_dd = pd.DataFrame(stand_dd)

y_pred_2 = model.predict(df_stand_dd)
y_pred_3 = y_pred_2 * 100
x_range = np.arange(1, len(y_pred_3) + 1)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_pred_3, label='predict')
plt.plot(x_range, dd['bis'], label='real')
plt.xlim(0,800)
plt.ylim(0,100)
plt.title('patient-1', fontsize=16)
plt.xlabel('data', fontsize=16)
plt.ylabel('bis', fontsize=16)
plt.legend()


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(dd['bis'], y_pred_3))


####### autokeras Classifier###########
import autokeras as ak
reg = ak.StructuredDataRegressor(max_trials=30, overwrite=True)
reg.fit(x=X_train, y=y_train,validation_data=(X_val, y_val), epochs=20)

# Evaluate the best model with testing data
loss, acc = reg.evaluate(X_val, y_val)

# Visualizing the best model
model = reg.export_model()
model.summary()

# Save the best model
model.save("ak_100-patient.h5", save_format="tf")

# Reload the model to make predictions
from tensorflow.keras.models import load_model
loaded_model = load_model("ak_100-patient.h5")
test = pd.read_csv('9_step3.csv')
test_dd = test.eeg
scaler = StandardScaler()
test_dd = pd.DataFrame(test_dd)
test_dd = scaler.fit_transform(test_dd)
test_dd = pd.DataFrame(test_dd)
pred = loaded_model.predict(test_dd)


ak_pred_2 = model.predict(pred)
ak_pred_3 = ak_pred_2 * 100
ak_range = np.arange(1, len(ak_pred_3) + 1)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_pred_3, label='predict')
plt.plot(x_range, dd['bis'], label='real')
plt.xlim(0,800)
plt.ylim(0,100)
plt.title('patient-1', fontsize=16)
plt.xlabel('data', fontsize=16)
plt.ylabel('bis', fontsize=16)
