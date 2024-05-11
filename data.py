#from matplotlib import Scalar
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Constants
N_STEPS = 50  # Look at the last 50 days to predict the next day
LOOKUP_STEP = 1  # Predict the next day
FUTURE_STEP = [1,2,3,4,5]

# Load data
ticker = 'AAPL'
data = yf.Ticker(ticker)
dataFrame = data.history(period='10y', interval='1d')

# Calculate technical indicators
dataFrame['RSI'] = talib.RSI(dataFrame['Close'], timeperiod=14)
dataFrame['SMA_50'] = talib.SMA(dataFrame['Close'], timeperiod=50)
dataFrame['EMA_50'] = talib.EMA(dataFrame['Close'], timeperiod=50)
dataFrame['MACD'], dataFrame['MACD_signal'], dataFrame['MACD_hist'] = talib.MACD(dataFrame['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
dataFrame['BB_upper'], dataFrame['BB_middle'], dataFrame['BB_lower'] = talib.BBANDS(dataFrame['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
dataFrame['ATR'] = talib.ATR(dataFrame['High'], dataFrame['Low'], dataFrame['Close'], timeperiod=14)

# Drop NaNs and unused columns
dataFrame.dropna(inplace=True)
dataFrame.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)

# Scaling features
scaler_feature = MinMaxScaler()
dataFrame_scaled = scaler_feature.fit_transform(dataFrame.drop(['Close'], axis=1))
scaler_target = MinMaxScaler()
dataFrame['Scaled_Close'] = scaler_target.fit_transform(dataFrame[['Close']])

# Define and compile the model #uncomment when making a new model
# model = Sequential([
#     LSTM(60, return_sequences=True, input_shape=(N_STEPS, X.shape[2])),
#     Dropout(0.3),
#     LSTM(120, return_sequences=False),
#     Dropout(0.3),
#     Dense(20),
#     Dense(1)
# ])
model = load_model('Apple_model.h5')
# model.compile(loss='mean_squared_error', optimizer='adam') #uncomment when needed to train new model
# model.fit(X, Y, epochs=80, batch_size=8, verbose=1)
# Save model for future use
#model.save('Apple_model.h5') #uncomment when saving

# Predict and append predictions
predicted_close_prices = []
for i in range(len(dataFrame) - N_STEPS):
    X_test = dataFrame_scaled[i:(i + N_STEPS)].reshape(1, N_STEPS, dataFrame_scaled.shape[1])
    pred_scaled = model.predict(X_test)
    pred_price = scaler_target.inverse_transform(pred_scaled)[0][0]
    predicted_close_prices.append(pred_price)

# Append predictions to the DataFrame
prediction_series = pd.Series(data=np.nan, index=dataFrame.index)
prediction_series[-len(predicted_close_prices):] = predicted_close_prices
dataFrame['Predicted_Close'] = prediction_series
# New prediction for the next 5 days
current_sequence = dataFrame_scaled[-N_STEPS:]
predictions = []
for step in FUTURE_STEP:
    # Reshape the sequence to match the input shape of the model: (batch_size, sequence_length, num_features)
    current_sequence = current_sequence.reshape(1, N_STEPS, -1)
    
    # Predict the next time step
    predicted_scaled = model.predict(current_sequence)
    predicted_price = scaler_target.inverse_transform(predicted_scaled)
    predictions.append(round(float(predicted_price), 2))
    predicted_scaled = predicted_scaled.reshape(1, 1, -1)
    # Update the current_sequence with the new predicted value:
    # 1. Remove the oldest time step from the sequence
    # 2. Append the new predicted_scaled value (reshape as needed to fit in the sequence)
    current_sequence[:, :-1, 0] = current_sequence[:, 1:, 0]
    # Insert the predicted value at the end of the sequence for the next prediction
    current_sequence[:, -1, 0] = predicted_scaled[:, 0, 0]
print("Next 5 days: ", predictions)
# Plotting
plt.figure(figsize=(14, 7))
plt.plot(dataFrame['Close'], label='Actual Closing Prices')
plt.plot(dataFrame['Predicted_Close'], label='Predicted Closing Prices', linestyle='--')
plt.title('Daily Stock Price Prediction - Validation')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

