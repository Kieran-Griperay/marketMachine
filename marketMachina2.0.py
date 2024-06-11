import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import numpy as np
import pandas as pd
import talib
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import datetime
import concurrent.futures
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set memory growth to avoid using all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
week_number = now.isocalendar().week
year_number = now.isocalendar().year
month_number = todays_date.month

# Load data
filtered_symbols_df = pd.read_csv('Stock Filtration/Backtesting/Results/Best_Stocks_Today.csv')
tickers = filtered_symbols_df['Ticker'].tolist()
workers = len(tickers)

dfs = {}
for t in tickers:
    data = yf.Ticker(t)
    df = data.history(period='8y', interval='1d')
    dfs[t] = df

# Parallel data download
def download_data(ticker):
    data = yf.Ticker(ticker)
    return data.history(period='8y', interval='1d')

# Use ThreadPoolExecutor for parallel data fetching
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(download_data, ticker): ticker for ticker in tickers}
    for future in as_completed(futures):
        ticker = futures[future]
        try:
            data = future.result()
            dfs[ticker] = data
        except Exception as exc:
            print(f"{ticker} generated an exception: {exc}")

def get_current_price(ticker):
    data = yf.Ticker(ticker)
    intraday_data = data.history(period='1d', interval='1m')
    if not intraday_data.empty:
        return intraday_data['Close'].iloc[-1]
    else:
        return None

@lru_cache(maxsize=128)
def get_yesterday_price(tick):
    ticker = yf.Ticker(tick)
    historical_data = ticker.history(period='2d')
    return historical_data['Close'].iloc[-2]

def add_technical_indicators(df):
    df['RSI'] = talib.RSI(df['Close'], timeperiod=28)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    df['EMA_90'] = talib.EMA(df['Close'], timeperiod = 90) 
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_middle'] = df['BB_middle'] * 1.75
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['Beta'] = talib.BETA(df['High'], df['Low'], timeperiod=10)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df.drop(columns=['Low', 'High', 'Stock Splits', 'BB_upper', 'BB_lower', 'Volume'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df

def create_dataset(X, Y, n_steps):
    Xs, Ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:(i + n_steps)])
        Ys.append(Y[i + n_steps])
    return np.array(Xs), np.array(Ys)

def showNewModelPredictions(dfs, ticker, model, scalers):
    base_directory = "plots/"
    df = dfs[ticker]
    scaler_feature, scaler_target = scalers
    fig_name = f"{year_number}_{ticker}_"
    columns_to_scale = df.drop(['Close', 'Predicted_Close'], axis=1, errors='ignore')
    features = scaler_feature.transform(columns_to_scale)
    target = scaler_target.transform(df[['Close']])
    try:
        pathy = str(base_directory) + str(fig_name) + "predictions.png"
        img = mpimg.imread(pathy)
    except (IOError, FileNotFoundError):
        predicted_close_prices = []
        for i in range(len(df) - N_STEPS):
            if len(df) < N_STEPS:
                print(f"Not enough data to predict {ticker}. Need at least N_STEPS entries.")
                continue
            X_test = features[i:(i + N_STEPS)].reshape(1, N_STEPS, features.shape[1])
            pred_scaled = model.predict(X_test, verbose=0)
            pred_price = scaler_target.inverse_transform(pred_scaled)[0][0]
            predicted_close_prices.append(pred_price)

        prediction_series = pd.Series(data=np.nan, index=df.index)
        prediction_series[-len(predicted_close_prices):] = predicted_close_prices
        df['Predicted_Close'] = prediction_series

        plt.figure(figsize=(14, 7))
        plt.plot(df['Close'], label='Actual Closing Prices')
        plt.plot(df.index[-len(predicted_close_prices):], predicted_close_prices, label='Predicted Closing Prices', linestyle='--')
        plt.title(f'Stock Price Prediction vs Validation for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.savefig(os.path.join(base_directory, f"{fig_name}predictions.png"))
        plt.close()

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
    features = scaler_feature.fit_transform(data.drop(['Close'], axis=1))
    target = scaler_target.fit_transform(data[['Close']].values)
    X, Y = create_dataset(features, target.ravel(), n_steps)

    if len(X) < 10:
        raise ValueError("Not enough data for training and validation.")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    batch_size = min(8, len(X_train))
    model = Sequential([
        LSTM(60, return_sequences=True, input_shape=(n_steps, X_train.shape[2])),
        Dropout(0.3),
        LSTM(120, return_sequences=False),
        Dropout(0.3),
        Dense(20),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, Y_train, epochs=35, batch_size=batch_size, verbose=1,
                        validation_data=(X_val, Y_val), callbacks=[early_stopping])

    val_loss = history.history.get('val_loss', [None])[-1]
    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(Y_val, val_predictions)
    mse = mean_squared_error(Y_val, val_predictions)
    rmse = np.sqrt(mse)
    
    with open('model_metrics.txt', 'a') as f:
        f.write(f"Ticker: {ticker} | Month: {month_number} | Val Loss: {val_loss} | MAE: {mae} | MSE: {mse} | RMSE: {rmse}\n")

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
    predicted_increase = calculate_percentage_increase(ticker, price, predicted_price)
    return ticker, predicted_increase

percentage_increase = {}
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(train_model_for_ticker, ticker, df): ticker for ticker, df in dfs.items()}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                if result:
                    ticker, increase = result
                    if increase is not None:
                        percentage_increase[ticker] = increase
                    else:
                        print(f"Training failed for ticker {ticker}")
            except concurrent.futures.TimeoutError:
                print(f"Timeout for ticker {ticker}")
            except Exception as exc:
                print(f"{ticker} generated an exception: {exc}")

    sorted_percentage_increase_map = sorted(percentage_increase.items(), key=lambda item: item[1], reverse=True)

    for pair in sorted_percentage_increase_map:
        print(f"Stock: {pair[0]}, Daily {'Increase' if pair[1] > 0 else 'Decrease'}: {round(pair[1], 3)}%")

    for i in range(min(3, len(sorted_percentage_increase_map))):
        tic = sorted_percentage_increase_map[i][0]
        showNewModelPredictions(dfs, tic, models[tic], scalers[tic])
