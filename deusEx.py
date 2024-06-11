from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
import os
from functools import lru_cache
import datetime
import concurrent.futures
import Stock_Filtration as sf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
N_STEPS = 50
LOOKUP_STEP = 1
models = {}
LOOKUP_STEPS = [5, 10, 20, 30]

scalers = {}
tf.get_logger().setLevel('ERROR')
# Determine the current week number
now = datetime.datetime.now()
todays_date = datetime.date.today() 
week_number = now.isocalendar().week
year_number = now.isocalendar().year
month_number = todays_date.month

# Load data
# filtered_symbols_df = pd.read_csv('filtered_symbols.csv')
# tickers = filtered_symbols_df['symbol'].tolist()
# workers = len(tickers) 
tickers = ["AAPL", "TSM", "ABBV", "BABA", "GOOGL", "AMZN", "ENPH"]
workers = min(os.cpu_count(), len(tickers) * len(LOOKUP_STEPS))


dfs = {}
for t in tickers:
    data = yf.Ticker(t)
    df = data.history(period='10y', interval='1d')
    dfs[t] = df

# Parallel data download
def download_data(ticker):
    data = yf.Ticker(ticker)
    return data.history(period='10y', interval='1d')

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

# Function to calculate technical indicators
def add_technical_indicators(df):
    """
    Calculate and append technical indicators to the DataFrame.
    Parameters:
    df (DataFrame): The input DataFrame containing stock prices.
    Returns:
    DataFrame: The DataFrame with added technical indicators.
    """
    df['Average_Price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['RSI'] = talib.RSI(df['Close'], timeperiod=28)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)*1.5 # weight
    df['EMA_90'] = talib.EMA(df['Close'], timeperiod = 90) * 1.5
    df['EMA_200'] = talib.EMA(df['Close'], timeperiod = 200)*1.5
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_middle'] = df['BB_middle'] * 1.75
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['Beta'] = talib.BETA(df['High'], df['Low'], timeperiod=10)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df.drop(columns=['Low', 'High', 'Stock Splits', 'BB_upper', 'BB_lower', 'Volume'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df


def create_dataset(X, Y, n_steps, lookup_step):
    """
    Creates dataset for LSTM model training.
    Parameters:
    X (array): Input features.
    Y (array): Target output.
    n_steps (int): Number of steps/lookback period.
    lookup_step (int): Number of steps to look ahead.
    Returns:
    tuple: Tuple containing feature and target datasets.
    """
    Xs, Ys = [], []
    for i in range(len(X) - n_steps - lookup_step):
        Xs.append(X[i:(i + n_steps)])
        Ys.append(Y[i + n_steps + lookup_step])
    return np.array(Xs), np.array(Ys)


def showNewModelPredictions(dfs, ticker, model, scalers):
    # Base directory to save plots
    base_directory = "plots/"

    df = dfs[ticker]
    scaler_feature, scaler_target = scalers
    fig_name = f"{year_number}_{ticker}_"

    # Ensure scalers are fit even if just loading models
    columns_to_scale = df.drop(['Close', 'Predicted_Close'], axis=1, errors='ignore')
    features = scaler_feature.transform(columns_to_scale)
    target = scaler_target.transform(df[['Close']])
    try:
        pathy = str(base_directory) + str(fig_name) + "predictions.png"
        img = mpimg.imread(pathy)
    except (IOError, FileNotFoundError):
        # Predicting past using features
        predicted_close_prices = []
        for i in range(len(df) - N_STEPS):
            if len(df) < N_STEPS:
                print(f"Not enough data to predict {ticker}. Need at least N_STEPS entries.")
                continue
            X_test = features[i:(i + N_STEPS)].reshape(1, N_STEPS, features.shape[1])
            pred_scaled = model.predict(X_test, verbose=0)
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
        plt.savefig(os.path.join(base_directory, f"{fig_name}predictions.png"))
        plt.close()

def calculate_percentage_increase(ticker, prix, predicted_closing):
    """
    Calculates the predicted percentage increase for the chosen stocks, return the best ones.
    Parameters: 
    ticker (str): used for the accessing of dfs dataframe.
    predicted closing price (ndarray): will be used for the predicted percentage increase
    Returns: 
    float: representing the increase or decrease of the stock
    """
    if not predicted_closing.size:  # Check if predicted_price is empty
        raise ValueError("Predicted price is empty")
    predicted_value = predicted_closing[0, 0]
    predicted_percent = (predicted_value - prix) / prix
    return predicted_percent * 100

def check_data_consistency(data_frames):
    """
    Checks the consistency of feature and row counts across multiple data frames.
    Parameters:
    data_frames (dict): A dictionary with tickers as keys and pandas DataFrame as values.
    Returns:
    dict: A dictionary reporting the number of features and rows for each ticker and any inconsistencies.
    """
    info_dict = {}
    standard_features = None
    standard_rows = None
    inconsistencies = []

    for ticker, df in data_frames.items():
        features = df.shape[1]
        rows = df.shape[0]
        info_dict[ticker] = {'Features': features, 'Rows': rows}
    return info_dict

def build_and_train_model(ticker, data, n_steps, lookup_step, test_size=0.1, random_state=42):
    """
    Builds and trains LSTM model.
    Parameters:
    ticker (str): The ticker symbol.
    data (DataFrame): The DataFrame containing training data.
    n_steps (int): Number of timesteps to look back.
    lookup_step (int): Number of steps to look ahead.
    Returns:
    tuple: Model and scaler objects, validation loss.
    """
    K.clear_session()
    scaler_feature = MinMaxScaler()
    scaler_target = MinMaxScaler()
    features = scaler_feature.fit_transform(data.drop(['Close'], axis=1))
    target = scaler_target.fit_transform(data[['Close']].values)
    X, Y = create_dataset(features, target.ravel(), n_steps, lookup_step)

    if len(X) < 10:
        raise ValueError("Not enough data for training and validation.")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    batch_size = min(8, len(X_train))  # Ensure batch size does not exceed the number of training samples
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
    history = model.fit(X_train, Y_train, epochs=40, batch_size=batch_size, verbose=1,
                        validation_data=(X_val, Y_val), callbacks=[early_stopping])

    val_loss = history.history.get('val_loss', [None])[-1]
    # Calculate evaluation metrics
    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(Y_val, val_predictions)
    mse = mean_squared_error(Y_val, val_predictions)
    rmse = np.sqrt(mse)
    
    # Save metrics to a text file
    with open('model_metrics.txt', 'a') as f:
        f.write(f"Ticker: {ticker} | Month: {month_number} | Lookup Step: {lookup_step} | Val Loss: {val_loss} | MAE: {mae} | MSE: {mse} | RMSE: {rmse} Model: | DeusExMachina \n")

    return model, scaler_feature, scaler_target, val_loss, mae, mse, rmse


# Prepare data with technical indicators
for ticker, df in dfs.items():
    dfs[ticker] = add_technical_indicators(df)

# Add fundamental indicators and recommendations to each DataFsrame
# for ticker, df in dfs.items():
#     dfs[ticker] = add_fundamental_indicators(df, ticker)
# Model training and prediction
get_action = lambda predicted_price, current_price: "BUY" if predicted_price > current_price + current_price * .005 else "HOLD"

def process_ticker(ticker, predicted_price, lookup_step):
    current_price = get_current_price(ticker)
    action = get_action(predicted_price, current_price)
    calculated_increase = calculate_percentage_increase(ticker, current_price, predicted_price)
    print(f"Stock: {ticker} | Lookup Step: {lookup_step} | Predicted Close Price: {predicted_price} | Current Price: {round(current_price, 5)} | Percent change: {round(calculated_increase, 4)}% | {action}")

percentage_increase = {}

def train_model_for_ticker(ticker, df):
    models[ticker] = {}
    scalers[ticker] = {}
    for lookup_step in LOOKUP_STEPS:
        weekly_model_path = f"models/deus_{year_number}_{month_number}_{ticker}_model_{lookup_step}.h5"
        scaler_feature_path = f"scalers/deus_{year_number}_{month_number}_{ticker}_scaler_feature_{lookup_step}.pkl"
        scaler_target_path = f"scalers/deus_{year_number}_{month_number}_{ticker}_scaler_target_{lookup_step}.pkl"
        try:
            model = load_model(weekly_model_path)
            scaler_feature = joblib.load(scaler_feature_path)
            scaler_target = joblib.load(scaler_target_path)
        except (IOError, FileNotFoundError):
            print(f"Training model for {ticker} with lookup step {lookup_step}")
            model, scaler_feature, scaler_target, val_loss, mae, mse, rmse = build_and_train_model(ticker, df, N_STEPS, lookup_step)
            model.save(weekly_model_path)
            joblib.dump(scaler_feature, scaler_feature_path)
            joblib.dump(scaler_target, scaler_target_path)
            print(f"Ticker: {ticker} | Lookup Step: {lookup_step} | Val Loss: {val_loss} | MAE: {mae} | MSE: {mse} | RMSE: {rmse}")
        models[ticker][lookup_step] = model
        scalers[ticker][lookup_step] = (scaler_feature, scaler_target)
        
        features = scaler_feature.transform(df.drop(['Close'], axis=1))
        current_sequence = features[-N_STEPS:].reshape(1, N_STEPS, features.shape[1])
        K.clear_session()
        predicted_scaled = model.predict(current_sequence, verbose=0)
        predicted_price = scaler_target.inverse_transform(predicted_scaled)
        process_ticker(ticker, predicted_price, lookup_step)
        price = get_yesterday_price(ticker)
        predicted_increase = calculate_percentage_increase(ticker, price, predicted_price)
        percentage_increase[f"{ticker}_{lookup_step}"] = predicted_increase


if __name__ == '__main__':
    percentage_increase = {}
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
        ticker, lookup_step = pair[0].rsplit('_', 1)
        print(f"Stock: {ticker}, Lookup Step: {lookup_step}, Daily {'Increase' if pair[1] > 0 else 'Decrease'}: {round(pair[1], 3)}%")

    for i in range(min(3, len(sorted_percentage_increase_map))):
        ticker, lookup_step = sorted_percentage_increase_map[i][0].rsplit('_', 1)
        showNewModelPredictions(dfs, ticker, models[ticker][int(lookup_step)], scalers[ticker][int(lookup_step)])
