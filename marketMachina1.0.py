import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import joblib
import datetime

# Constants
N_STEPS = 50
LOOKUP_STEP = 1
FUTURE_STEP = [1, 2, 3, 4, 5]

# Determine the current week number
now = datetime.datetime.now()
week_number = now.isocalendar().week

# Load data
tickers = ['AAPL', 'NVDA', 'GOOG', 'PGR', 'SPOT', 'AMZN', 'TGT', 'TSM', 'TSLA', 'NTNX']
dfs = {}
for t in tickers:
    data = yf.Ticker(t)
    df = data.history(period='10y', interval='1d')
    dfs[t] = df

# Function to calculate technical indicators
def add_technical_indicators(df):
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df.dropna(inplace=True)
    return df

# Function to create LSTM datasets
def create_dataset(X, Y, n_steps):
    Xs, Ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:(i + n_steps)])
        Ys.append(Y[i + n_steps])
    return np.array(Xs), np.array(Ys)

def showNewModelPredictions(dfs, models, scalers, week_number):
    import os

    # Base directory to save plots
    base_directory = "plots/"
    for ticker, model in models.items():
        df = dfs[ticker]
        scaler_feature, scaler_target = scalers[ticker]
        fig_name = f"{week_number}_{ticker}"
        # Ensure scalers are fit even if just loading models
        features = scaler_feature.transform(df.drop(['Close'], axis=1))
        target = scaler_target.transform(df[['Close']])

        # Predicting past using features
        predicted_close_prices = []
        for i in range(len(df) - N_STEPS):
            X_test = features[i:(i + N_STEPS)].reshape(1, N_STEPS, features.shape[1])
            pred_scaled = model.predict(X_test)
            pred_price = scaler_target.inverse_transform(pred_scaled)[0][0]
            predicted_close_prices.append(pred_price)

        # Append predictions to DataFrame
        prediction_series = pd.Series(data=np.nan, index=df.index)
        prediction_series[-len(predicted_close_prices):] = predicted_close_prices
        df['Predicted_Close'] = prediction_series

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(df['Close'], label='Actual Closing Prices')
        plt.plot(df.index[-len(predicted_close_prices):], predicted_close_prices, label='Predicted Closing Prices', linestyle='--')
        plt.title(f'Stock Price Prediction vs Validation for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        # Save the figure
        plt.savefig(os.path.join(base_directory, f"{fig_name}_predictions.png"))
        plt.close()  # Close the plot to free up memory



# Function to build and train LSTM models
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
    model.fit(X, Y, epochs=50, batch_size=8, verbose=1)
    return model, scaler_feature, scaler_target

# Prepare data with technical indicators
for ticker, df in dfs.items():
    dfs[ticker] = add_technical_indicators(df)

# Model training and prediction
models = {}
scalers = {}

for ticker, df in dfs.items():

    weekly_model_path = f"models/{week_number}{ticker}_model.h5"
    scaler_feature_path = f"scalers/{ticker}_scaler_feature.pkl"
    scaler_target_path = f"scalers/{ticker}_scaler_target.pkl"

    try:
        model = load_model(weekly_model_path)
        scaler_feature = joblib.load(scaler_feature_path)
        scaler_target = joblib.load(scaler_target_path)
    except (IOError, FileNotFoundError):   # If not existent, build and train a new model, !!! ONLY EXECUTES WHEN NO OLD MODELS ARE FOUND, CALLS BUILD MODEL FUNCTION !!!
        print(f"Training model for {ticker}")
        model, scaler_feature, scaler_target = build_and_train_model(df, N_STEPS)
        model.save(weekly_model_path)
        joblib.dump(scaler_feature, scaler_feature_path)
        joblib.dump(scaler_target, scaler_target_path)
        #showNewModelPredictions(df, model, scaler_feature, week_number)

    models[ticker] = model
    scalers[ticker] = (scaler_feature, scaler_target)

    # Ensure proper scaling and input shape
    features = scaler_feature.transform(df.drop(['Close'], axis=1))
    current_sequence = features[-N_STEPS:].reshape(1, N_STEPS, features.shape[1])
    if current_sequence.shape[2] != 16:  # Assuming the model was trained with 16 features
        raise ValueError(f"Incorrect number of features. Expected 16, found {current_sequence.shape[2]}")

    predicted_scaled = model.predict(current_sequence)
    predicted_price = scaler_target.inverse_transform(predicted_scaled)
    print(f"Next closing price for {ticker}: {predicted_price}")

    if now.weekday() == 0:  # Check if today is Monday
        showNewModelPredictions(dfs, models, scalers, week_number)

