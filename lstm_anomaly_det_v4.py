# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:25:33 2023

@author: Basant Kumar
"""

# import libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
# %matplotlib inline

from numpy.random import seed
# from tensorflow import set_random_seed
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)


from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
#%%


df = pd.read_csv("C:/Lehigh/Rick- Anomaly Detection/CNN-KF/CNN-KF/Clean_data.csv", index_col='Epoch')

df = df.rename({'InVehicle_Longitudinal_Speed':'inveh_long_spd', 'GPS_Speed': 'gps_speed', 'InVehicle_Longitudinal_Accel': 'inveh_long_acc'}, axis = 1)

df = df.dropna()
# df.Device.value_counts()
# # Devices  Counts
# # 17102    6596065
# # 17103    5741834
# # 17101    4803795
# # 10582     690542
# # 13101     621452

# # Filter for a unique device

# df_1dev = df[df['Device'] == 17102]

# df_1dev = df_1dev[['Trip', 'GpsSpeed', 'GpsHeading', 'Longitude', 'Altitude']]

# df_1dev_1trip = df_1dev[df_1dev[ 'Trip'] == 36]

# df_1dev_1trip = df_1dev_1trip[['GpsSpeed', 'GpsHeading', 'Longitude', 'Altitude']]

# plt.plot(df_1dev_1trip['GpsSpeed'])

#%%

train_size = int(len(df) * 0.6)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

#%%

fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train['inveh_long_spd'], label='inveh_long_spd', color='blue', animated = True, linewidth=1)
# ax.plot(train['GpsHeading'], label='GpsHeading', color='red', animated = True, linewidth=1)
# ax.plot(train['Latitude'], label='Latitude', color='green', animated = True, linewidth=1)
ax.plot(train['gps_speed'], label='gps_speed', color='black', animated = True, linewidth=1)
ax.plot(train['inveh_long_acc'], label='inveh_long_acc', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Sensor Training Data', fontsize=16)
plt.show()

#%%

fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(test['inveh_long_spd'], label='inveh_long_spd', color='blue', animated = True, linewidth=1)
# ax.plot(test['GpsHeading'], label='GpsHeading', color='red', animated = True, linewidth=1)
# ax.plot(test['Latitude'], label='Latitude', color='green', animated = True, linewidth=1)
ax.plot(test['gps_speed'], label='gps_speed', color='black', animated = True, linewidth=1)
ax.plot(test['inveh_long_acc'], label='inveh_long_acc', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Sensor Testing Data', fontsize=16)
plt.show()

#%%

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(df)
X_train = scaler.transform(train)
X_test = scaler.transform(test)
#%%

# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)

#%%

# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model
#%%

# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mse')
model.summary()

#%%
# fit the model to the data
nb_epochs = 30
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.1).history

#%%

# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mse)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

#%%
import seaborn as sns
# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
Loss_mse = np.mean(np.abs(X_pred-Xtrain), axis = 1)

sns.distplot(Loss_mse, bins=50, kde= True)

#%%

threshold = np.percentile(Loss_mse, 99)

#%%

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mse'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = threshold
scored['Anomaly'] = scored['Loss_mse'] > scored['Threshold']
scored.head()

#%%

# plot bearing failure time plot
scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])