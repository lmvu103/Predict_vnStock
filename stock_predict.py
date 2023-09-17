from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import keras.callbacks
from sklearn.preprocessing import MinMaxScaler
#Importing necessary libraries
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from vnstock import * #import all functions

# convert an array of values into a data_set matrix def
def create_data_set(_data_set, _look_back=1):
    data_x, data_y = [], []
    for i in range(len(_data_set) - _look_back - 1):
        a = _data_set[i:(i + _look_back), 0]
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0])
    return np.array(data_x), np.array(data_y)

START = "2010-01-01"
TODAY = date.today().strftime('%Y-%m-%d')

st.write("""
# Viet Nam Stock Price Prediction App
""")

selected_stock = st.text_input("Please input stock code:")

if selected_stock == "":
    st.write("Please input the stock code")
else:
    df=stock_historical_data(selected_stock, START, TODAY, "1D")
    st.subheader("1.1 Raw data")
    st.write(df)

    st.subheader("1.2 History stock data")

    fig = plt.figure(figsize=(15, 5))
    plt.title('Close Price History ' + selected_stock)
    plt.plot(df['close'])
    plt.plot(df['close'], label='Close Price')
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Close Price vnd)', fontsize=20)
    st.pyplot(fig)
    
    method_predict = st.selectbox("Select the method:", ("Prophet", "LSTM"))
    n_days = st.slider("Days of prediction:", 30, 60)
    period = n_days
    
    st.subheader("1.3 Predict price stock")

    if (st.button('Press to predict Stock')):
        if method_predict == "Prophet":
            #st.write('Please choice LSTM')
            df_train = pd.DataFrame()
            df_train['ds'] = df['time']
            df_train['y'] = df['close']    
            st.write(df_train)            
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)
            st.subheader('Forecast data')
            st.write(forecast.tail())
            st.write('Forecast data')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
            st.write('Forecast Component')
            fig2 = m.plot_components(forecast)
            st.write(fig2)        
        
        elif method_predict == "LSTM":
            data = df['close']
            data=data.reset_index()
            dataset = data.values
            st.subheader('Forecast data')
            st.write(dataset)
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(dataset)
            train_size = int(len(scaled_data) * 0.70)
            test_size = len(scaled_data) - train_size
            train, test = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]
            # reshape into X=t and Y=t+1
            look_back =90
            X_train,Y_train,X_test,Y_test = [],[],[],[]
            X_train,Y_train=create_data_set(train,look_back)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test,Y_test=create_data_set(test,look_back)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # create and fit the LSTM network regressor = Sequential() 
            regressor = Sequential()
            regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units = 50, return_sequences = True))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units = 50))
            regressor.add(Dropout(0.2))
            regressor.add(Dense(units = 1))
            regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5)
            history =regressor.fit(X_train, Y_train, epochs = 5, batch_size = 32,validation_data=(X_test, Y_test), callbacks=[reduce_lr],shuffle=False)
            train_predict = regressor.predict(X_train)
            test_predict = regressor.predict(X_test)
            st.write(train_predict)
            st.write(test_predict)
            train_predict = scaler.inverse_transform(train_predict)
            Y_train = scaler.inverse_transform([Y_train])
            test_predict = scaler.inverse_transform(test_predict)
            Y_test = scaler.inverse_transform([Y_test])
            # visualization
            st.subheader("Model loss")
            fig3 = plt.figure(figsize=(15, 5))
            plt.title('Close Price History ' + selected_stock)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Test Loss')
            plt.ylabel('loss')
            plt.xlabel('epochs')
            st.pyplot(fig3)
            #creatinf testing dataset
            test_data = scaled_data[training_data_len - 60: , :]
            #creating x_test and y_tets datasets
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range (60, len(test_data)):
                x_test.append(test_data[i -60:i, 0])
            #converting data to numpy array
            x_test = np.array(x_test)

            #reshape data
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            #get predicted price values
            predictions = regressor.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            
            # plot the data
            train = data[:train_size]
            valid = data[train_size:]
            valid['Predictions'] = predictions
            
            # visualization
            st.subheader("Predict history price stock " + method_predict)
            fig4 = plt.figure(figsize=(15, 5))
            plt.title('Close Price History ' + selected_stock)
            plt.plot(train['close'])
            plt.plot(valid[['close', 'Predictions']])
            plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
            plt.xlabel('Date', fontsize=20)
            plt.ylabel('Close Price vnd)', fontsize=20)
            st.pyplot(fig4)
            dataset_test = data[- period:].values
            inputs = dataset_test
            inputs = inputs.reshape(-1, 1)
            inputs = scaler.transform(inputs)
            i = 0
            while i < period:
                X_test = []
                no_of_sample = len(inputs)
                # Lay du lieu cuoi cung
                X_test.append(inputs[no_of_sample - period:no_of_sample, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                # Du doan gia
                predicted_stock_price = regressor.predict(X_test)
                # chuyen gia tu khoang (0,1) thanh gia that
                predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
                dataset_test = np.append(dataset_test, predicted_stock_price[0])
                inputs = dataset_test
                inputs = inputs.reshape(-1, 1)
                inputs = scaler.transform(inputs)
                i = i + 1
                # visualization
                dataset_pre = np.append(data['close'], dataset_test[period:])
                st.subheader("Predict future price stock next " + str(period))
                fig5 = plt.figure(figsize=(15, 5))
                plt.plot(dataset_pre)
                plt.plot(dataset[:])
                plt.legend(['Prediction', 'History'], loc='upper right')
                st.pyplot(fig5)           


                 
