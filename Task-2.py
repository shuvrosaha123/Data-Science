import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load historical stock price data
stock_data = yf.download('AAPL', start='2010-01-01', end='2022-01-01', progress=False)

# Take only the 'Close' column
stock_prices = stock_data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_stock_prices = scaler.fit_transform(stock_prices)

# Split data into training and testing sets
train_size = int(len(scaled_stock_prices) * 0.8)
test_size = len(scaled_stock_prices) - train_size
train_data, test_data = scaled_stock_prices[0:train_size, :], scaled_stock_prices[train_size:len(scaled_stock_prices), :]

# Function to create datasets
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Reshape into X=t and Y=t+1
time_step = 100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64, verbose=1)

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(stock_prices, label='Actual Stock Price')
plt.plot(np.arange(time_step, len(train_predictions) + time_step), train_predictions, label='Training Predictions')
plt.plot(np.arange(time_step + len(train_predictions), len(stock_prices)), test_predictions, label='Test Predictions')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction using LSTM')
plt.legend()
plt.show()
