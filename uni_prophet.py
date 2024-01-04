
# Import required libraries
import pandas as pd

# Read and Plot data
all = pd.read_excel(open('final_bit.xlsx', 'rb')) 
all.plot(x='Date', y='Close')
all = all.iloc[1950:]


# Convert monthly data to datatime format
df = pd.DataFrame()
df['Date'] = pd.to_datetime(all['Date'])
df['y'] = all['Close']

# prophet required column names - time index: 'ds', values: 'y' 
df.columns = ['ds', 'y']

# Modeling
from prophet import Prophet

model = Prophet()
model.fit(df)

test = pd.read_excel(open('test.xlsx', 'rb')) 
df_test = test
df_test2 = pd.DataFrame()
df_test2['ds'] = pd.to_datetime(df_test['Date'])
df_test2['y'] = df_test['Close']

forecast = model.predict(df_test2)
forecast_close = forecast[['ds', 'yhat']]
final_df = pd.concat((forecast_close['yhat'], df_test2), axis=1)

# Plot actual & predicted comparison
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.plot(final_df['ds'], final_df['y'], color='red', label = 'actual')
plt.plot(final_df['ds'], final_df['yhat'], color='blue', label = 'forecast')
import numpy as np

## Evaluation
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred)) # mean absolute error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mape

mae, mape = mean_absolute_percentage_error(final_df['y'], final_df['yhat'])
print('MAE = ', mae, '; MAPE = ', mape)
    
    
####### Save and load moldel  #######
import pickle
with open('Univariate.pckl', 'wb') as fout:
    pickle.dump(model, fout)

with open('model.pckl', 'rb') as fin:
    m = pickle.load(fin)
    