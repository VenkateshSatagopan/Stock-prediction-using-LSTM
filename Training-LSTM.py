# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:11:12 2019

@author: venkatesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


#Data import
Training=False  # Setting variable to choose whether script needs to run training part or not
#print(os.listdir())
train_dataset=pd.read_csv('train-data-DTAG-CSV.csv')
train_data=train_dataset.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
scaled_train_data=sc.fit_transform(train_data)

if Training:
 # Create a data structure with 60 timestamps and 1 output
 X_train=[]
 y_train=[]

 for i in range(60,len(scaled_train_data)):
    X_train.append(scaled_train_data[i-60:i,0])
    y_train.append(scaled_train_data[i,0])
    
 X_train,y_train=np.array(X_train),np.array(y_train)

 #Reshaping
 X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


 #Importing the Keras libraries and packages
 from keras.models import Sequential
 from keras.layers import Dense
 from keras.layers import LSTM
 from keras.layers import Dropout


 #Create Sequential model
 model=Sequential()

 #Add first LSTM layer + Dropout
 model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
 model.add(Dropout(0.2))

 # Adding a second LSTM layer and some Dropout regularisation
 model.add(LSTM(units = 50, return_sequences = True))
 model.add(Dropout(0.2))

 # Adding a third LSTM layer and some Dropout regularisation
 model.add(LSTM(units = 50, return_sequences = True))
 model.add(Dropout(0.2))

 # Adding a fourth LSTM layer and some Dropout regularisation
 model.add(LSTM(units = 50))
 model.add(Dropout(0.2))

 # Adding the output layer
 model.add(Dense(units = 1))


 model.compile(optimizer='adam',loss='mean_squared_error')

 # Fitting the RNN to the Training set
 model.fit(X_train, y_train, epochs = 100, batch_size = 32,verbose=10)

 model.save('final-model.h5')


# Part 3 - Making the predictions and visualising the results
from keras.models import load_model
model=load_model('final-model.h5')

# Getting the real stock price of 2019
dataset_test = pd.read_csv('test-data-DTAG-CSV.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2019
dataset_total = pd.concat((train_dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real DTAG Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted DTAG Stock Price')
plt.title('DTAG Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('DTAG Stock Price')
plt.legend()
plt.show()

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
