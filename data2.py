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
FUTURE_STEP = [1, 2, 3, 4, 5]

# Load data
tickers = ['AAPL', 'NVDA', 'GOOG']
dfs = {}
for t in tickers:
    data = yf.Ticker(t)
    df = data.history(period='10y', interval='1d')  # Fetch 10 years of daily data
    dfs[t] = df

#Technical indicators
def add_technical_indicators(df):
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    df['Annual_High'] = df['High'].rolling(window=252).max() #Annual High
    df['Annual_Low'] = df['Low'].rolling(window=252).min() #Annual Low
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df.dropna(inplace=True)
    return df

#Fundamental Indicators
info = data.info
df['Market_Cap'] = info.get('marketCap', None) #Market Cap
df['Enterprise_Value'] = info.get('enterpriseValue', None) #Enterprise Value
df['Forward_PE'] = info.get('forwardPE', None) #PE Ratio
df['PEG_Ratio'] = info.get('pegRatio', None) #PEG Ratio
df['Beta'] = info.get('beta', None) #Beta


#Prepare data
for ticker, df in dfs.items():
    dfs[ticker] = add_technical_indicators(df)

#Function to create dataset for LSTM
def create_dataset(X, Y, n_steps):
    Xs, Ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:(i + n_steps)])
        Ys.append(Y[i + n_steps])
    return np.array(Xs), np.array(Ys)

# Function to build and train model
def build_and_train_model(data, n_steps):
    scaler_feature = MinMaxScaler()
    scaler_target = MinMaxScaler()
    features = scaler_feature.fit_transform(data.drop(['Close'], axis=1))
    target = scaler_target.fit_transform(data[['Close']].values)
    X, Y = create_dataset(features, target.ravel(), n_steps)
    
    model = Sequential([
        LSTM(60, return_sequences=True, input_shape=(n_steps, X.shape[2])),
        Dropout(0.3),
        LSTM(120, return_sequences=False),
        Dropout(0.3),
        Dense(20),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=40, batch_size=8, verbose=1)
    return model, scaler_feature, scaler_target

# Train model for each ticker
models = {}
scalers = {}
for ticker, df in dfs.items():
    print(f"Training model for {ticker}")
    model, scaler_feature, scaler_target = build_and_train_model(df, N_STEPS)
    models[ticker] = model
    scalers[ticker] = (scaler_feature, scaler_target)
    model.save(f'{ticker}_model.h5')  # Save model

# Predict and visualize for each ticker
for ticker, model in models.items():
    df = dfs[ticker]
    scaler_feature, scaler_target = scalers[ticker]
    features = scaler_feature.transform(df.drop(['Close'], axis=1))
    predicted_close_prices = []
    for i in range(len(df) - N_STEPS):
        X_test = features[i:(i + N_STEPS)].reshape(1, N_STEPS, features.shape[1])
        pred_scaled = model.predict(X_test)
        pred_price = scaler_target.inverse_transform(pred_scaled)[0][0]
        predicted_close_prices.append(pred_price)
    
    prediction_series = pd.Series(data=np.nan, index=df.index)
    prediction_series[-len(predicted_close_prices):] = predicted_close_prices
    df['Predicted_Close'] = prediction_series
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Actual Closing Prices')
    plt.plot(df['Predicted_Close'], label='Predicted Closing Prices', linestyle='--')
    plt.title(f'Daily Stock Price Prediction - Validation for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()