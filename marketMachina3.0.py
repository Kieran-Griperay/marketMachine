import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from functools import lru_cache
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set mixed precision policy
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Ensure TensorFlow is using the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Constants
N_STEPS = 50
LOOKUP_STEP = 1
FUTURE_STEP = [1, 2, 3, 4, 5]
models = {}
scalers = {}
tf.get_logger().setLevel('ERROR')

# Determine the current week number
now = datetime.datetime.now()
todays_date = datetime.date.today() 
year_number, week_number, _ = now.isocalendar()
month_number = todays_date.month

# Load data
filtered_symbols_df = pd.read_csv('Stock Filtration/Backtesting/Results/Best_Stocks_Today.csv')
tickers = filtered_symbols_df['Ticker'].tolist()
tickers = tickers + ["AAPL", "TSM", "ABBV", "GOOGL", "AMZN"]#Look after these stocks
print(tickers)
# Download data with a fallback period
def download_data(ticker, primary_period='1y', fallback_period='max'):
    data = yf.Ticker(ticker)
    try:
        df = data.history(period=primary_period, interval='1d')
        if df.empty:
            raise ValueError(f"No data found for {ticker} with period {primary_period}.")
    except Exception as e:
        print(f"{ticker} generated an exception with period {primary_period}: {e}. Trying {fallback_period} period.")
        try:
            df = data.history(period=fallback_period, interval='1d')
            if df.empty:
                raise ValueError(f"No data found for {ticker} with period {fallback_period}.")
        except Exception as e:
            print(f"{ticker} generated an exception with period {fallback_period}: {e}. Skipping this ticker.")
            return None
    return df

# Sequential data fetching and processing
dfs = {}
for ticker in tickers:
    data = download_data(ticker)
    if data is not None:
        dfs[ticker] = data

def get_current_price(ticker):
    data = yf.Ticker(ticker)
    intraday_data = data.history(period='1d', interval='1m')
    if not intraday_data.empty:
        return intraday_data['Close'].iloc[-1]
    else:
        return None

@lru_cache(maxsize=128)
def get_yesterday_price(ticker):
    data = yf.Ticker(ticker)
    historical_data = data.history(period='5d')
    if len(historical_data) < 2:
        print(f"{ticker}: Not enough data for '3d' period.")
        return None
    return historical_data['Close'].iloc[-2]

def add_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=28).rsi()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['EMA_90'] = ta.trend.EMAIndicator(df['Close'], window=90).ema_indicator()
    
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg() * 1.75
    df['BB_lower'] = bb.bollinger_lband()

    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df.drop(columns=['Low', 'High', 'Stock Splits', 'BB_upper', 'BB_lower', 'Volume'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    
    return df

def create_dataset(X, Y, n_steps):
    Xs, Ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:(i + n_steps)])
        Ys.append(Y[i + n_steps])
    return np.array(Xs), np.array(Ys)

def calculate_percentage_increase(ticker, prix, predicted_closing):
    if not predicted_closing.size:
        raise ValueError("Predicted price is empty")
    predicted_value = predicted_closing[0, 0]
    predicted_percent = (predicted_value - prix) / prix
    return predicted_percent * 100

def check_data_consistency(data_frames):
    info_dict = {}
    for ticker, df in data_frames.items():
        features = df.shape[1]
        rows = df.shape[0]
        info_dict[ticker] = {'Features': features, 'Rows': rows}
    return info_dict

def build_and_train_model(ticker, data, n_steps, test_size=0.1, random_state=42):
    K.clear_session()
    
    scaler_feature = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    # Scale features and target
    features = scaler_feature.fit_transform(data.drop(['Close'], axis=1))
    target = scaler_target.fit_transform(data[['Close']].values)
    
    # Create dataset
    X, Y = create_dataset(features, target.ravel(), n_steps)
    
    if len(X) < 10:
        raise ValueError("Not enough data for training and validation.")
    
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    batch_size = min(64, len(X_train))  # Increase batch size to 64
    
    # Build the model
    model = Sequential([
        LSTM(60, return_sequences=True, input_shape=(n_steps, X_train.shape[2])),
        Dropout(0.3),
        LSTM(120, return_sequences=False),
        Dropout(0.3),
        Dense(20),
        Dense(1, dtype='float32')  # Ensure final layer outputs float32
    ])
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, Y_train, epochs=50, batch_size=batch_size, verbose=1,
                        validation_data=(X_val, Y_val), callbacks=[early_stopping])
    
    # Evaluate the model
    val_loss = history.history.get('val_loss', [None])[-1]
    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(Y_val, val_predictions)
    mse = mean_squared_error(Y_val, val_predictions)
    rmse = np.sqrt(mse)
    
    # Log the metrics
    with open('model_metrics.txt', 'a') as f:
        f.write(f"Ticker: {ticker} | Val Loss: {val_loss} | MAE: {mae} | MSE: {mse} | RMSE: {rmse}\n")
    
    return model, scaler_feature, scaler_target, val_loss, mae, mse, rmse

for ticker, df in dfs.items():
    dfs[ticker] = add_technical_indicators(df)

get_action = lambda predicted_price, current_price: "BUY" if predicted_price > current_price + current_price * .005 else "HOLD"

def process_ticker(ticker, predicted_price):
    current_price = get_current_price(ticker)
    action = get_action(predicted_price, current_price)
    calculated_increase = calculate_percentage_increase(ticker, current_price, predicted_price)
    print(f"Stock: {ticker} | Predicted Close Price: {predicted_price} | Current Price: {round(current_price,5)} | Percent change: {round(calculated_increase, 4)}%| {action}")

percentage_increase = {}

def train_model_for_ticker(ticker, df):
    weekly_model_path = f"models/{year_number}_{month_number}_{ticker}_model.h5"
    scaler_feature_path = f"scalers/{year_number}_{month_number}_{ticker}_scaler_feature.pkl"
    scaler_target_path = f"scalers/{year_number}_{month_number}_{ticker}_scaler_target.pkl"
    try:
        model = load_model(weekly_model_path)
        scaler_feature = joblib.load(scaler_feature_path)
        scaler_target = joblib.load(scaler_target_path)
    except (IOError, FileNotFoundError):
        print(f"Training model for {ticker}")
        model, scaler_feature, scaler_target, val_loss, mae, mse, rmse = build_and_train_model(ticker, df, N_STEPS)
        model.save(weekly_model_path)
        joblib.dump(scaler_feature, scaler_feature_path)
        joblib.dump(scaler_target, scaler_target_path)
        print(f"Ticker: {ticker} | Val Loss: {val_loss} | MAE: {mae} | MSE: {mse} | RMSE: {rmse}")
    models[ticker] = model
    scalers[ticker] = (scaler_feature, scaler_target)
    features = scaler_feature.transform(df.drop(['Close'], axis=1))
    current_sequence = features[-N_STEPS:].reshape(1, N_STEPS, features.shape[1])
    K.clear_session()
    predicted_scaled = model.predict(current_sequence, verbose=0)
    predicted_price = scaler_target.inverse_transform(predicted_scaled)
    process_ticker(ticker, predicted_price)
    price = get_yesterday_price(ticker)
    if price is not None:
        predicted_increase = calculate_percentage_increase(ticker, price, predicted_price)
        return ticker, predicted_increase
    else:
        return ticker, None

percentage_increase = {}
if __name__ == '__main__':
    for ticker, df in dfs.items():
        try:
            result = train_model_for_ticker(ticker, df)
            if result:
                ticker, increase = result
                if increase is not None:
                    percentage_increase[ticker] = increase
                else:
                    print(f"Training failed for ticker {ticker}")
        except Exception as exc:
            print(f"{ticker} generated an exception: {exc}")

    sorted_percentage_increase_map = sorted(percentage_increase.items(), key=lambda item: item[1], reverse=True)
    
    for pair in sorted_percentage_increase_map:
        print(f"Stock: {pair[0]}, Daily {'Increase' if pair[1] > 0 else 'Decrease'}: {round(pair[1], 3)}%")
    
    # Save the top 15 stocks to a CSV file
    top_15_stocks = sorted_percentage_increase_map[:15]
    top_15_df = pd.DataFrame(top_15_stocks, columns=['Ticker', 'Predicted_Percent_Increase'])
    top_15_df.to_csv('top_15_stocks.csv', index=False)
    print("Top 15 stocks saved to top_15_stocks.csv")
