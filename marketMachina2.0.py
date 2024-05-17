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
import joblib
from tensorflow.keras import backend as K
import os
import datetime
# Constants
N_STEPS = 50
LOOKUP_STEP = 1
FUTURE_STEP = [1, 2, 3, 4, 5]

# Determine the current week number
now = datetime.datetime.now()
week_number = now.isocalendar().week

# Load data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-A', 'BRK-B', 'JPM', 'JNJ', 'V', 'WMT', 'UNH', 'MA', 'PG', 'HD', 'PYPL', 'DIS', 'BABA', 'BAC', 'CMCSA', 'XOM', 'T', 'VZ', 'CRM', 'INTC', 'CSCO', 'NFLX', 'KO', 'PEP', 'ABT', 'ADBE', 'MRK', 'NKE', 'ACN', 'NVDA', 'PFE', 'CVX', 'MCD', 'ABBV', 'COST', 'WFC', 'DHR', 'AVGO', 'QCOM', 'NEE', 'TXN', 'UPS', 'TMUS']
dfs = {}
for t in tickers:
    data = yf.Ticker(t)
    df = data.history(period='10y', interval='1d')
    dfs[t] = df

# Function to calculate technical indicators
def add_technical_indicators(df):
    """
    Calculate and append technical indicators to the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing stock prices.

    Returns:
    DataFrame: The DataFrame with added technical indicators.
    """
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df.dropna(inplace=True)
    return df
ticker_info = {}
for ticker in tickers:
    ticker_info[ticker] = yf.Ticker(ticker).info
def add_fundamental_indicators(df, info):
    """
    Calculate and append fundamental indicators to the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing stock prices.

    Returns:
    DataFrame: The DataFrame with added fundamental indicators.
    """
    fundamental_indicators = {
        'Market_Cap': info.get('marketCap', None),
        'Enterprise_Value': info.get('enterpriseValue', None),
        'Forward_PE': info.get('forwardPE', None),
        'PEG_Ratio': info.get('pegRatio', None),
        'Beta': info.get('beta', None)
    }
    for name, indicator in fundamental_indicators.items():
        df[name] = indicator
    return df

# Function to create LSTM datasets
def create_dataset(X, Y, n_steps):
    """
    Creates dataset for LSTM model training.

    Parameters:
    X (array): Input features.
    Y (array): Target output.
    n_steps (int): Number of steps/lookback period.

    Returns:
    tuple: Tuple containing feature and target datasets.
    """
    Xs, Ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:(i + n_steps)])
        Ys.append(Y[i + n_steps])
    return np.array(Xs), np.array(Ys)
def showNewModelPredictions(dfs, models, scalers, week_number):
    
    # Base directory to save plots
    base_directory = "plots/"
    for ticker, model in models.items():
        df = dfs[ticker]
        scaler_feature, scaler_target = scalers[ticker]
        fig_name = f"{week_number}_{ticker}"
        # Ensure scalers are fit even if just loading models
        columns_to_scale = df.drop(['Close', 'Predicted_Close'], axis=1, errors='ignore')
        features = scaler_feature.transform(columns_to_scale)
        target = scaler_target.transform(df[['Close']])

        # Predicting past using features
        predicted_close_prices = []
        for i in range(len(df) - N_STEPS):
            if len(df) < N_STEPS:
                print(f"Not enough data to predict {ticker}. Need at least {N_STEPS} entries.")
                continue
            X_test = features[i:(i + N_STEPS)].reshape(1, N_STEPS, features.shape[1])
            pred_scaled = model.predict(X_test, verbose = 0)
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


def build_and_train_model(data, n_steps, test_size=0.1, random_state=42):
    """
    Builds and trains LSTM model.

    Parameters:
    data (DataFrame): The DataFrame containing training data.
    n_steps (int): Number of timesteps to look back.

    Returns:
    tuple: Model and scaler objects, validation loss.
    """
    K.clear_session()
    scaler_feature = MinMaxScaler()
    scaler_target = MinMaxScaler()
    features = scaler_feature.fit_transform(data.drop(['Close'], axis=1))
    target = scaler_target.fit_transform(data[['Close']].values)
    X, Y = create_dataset(features, target.ravel(), n_steps)

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
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_val shape:", Y_val.shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, Y_train, epochs=50, batch_size=batch_size, verbose=1,
                        validation_data=(X_val, Y_val), callbacks=[early_stopping])

    print(history.history.keys())
    val_loss = history.history.get('val_loss', [None])[-1]

    return model, scaler_feature, scaler_target, val_loss


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
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Expected data to be a pandas DataFrame, got {}".format(type(df)))
        #print("Type of data before calling build_and_train_model:", type(data))

        model, scaler_feature, scaler_target, val_loss = build_and_train_model(df, N_STEPS)
        model.save(weekly_model_path)
        print(f"Validation loss for {ticker}: {val_loss}")

        joblib.dump(scaler_feature, scaler_feature_path)
        joblib.dump(scaler_target, scaler_target_path)
        #showNewModelPredictions(df, model, scaler_feature, week_number)

    models[ticker] = model
    scalers[ticker] = (scaler_feature, scaler_target)

    # Ensure proper scaling and input shape
    features = scaler_feature.transform(df.drop(['Close'], axis=1))
    current_sequence = features[-N_STEPS:].reshape(1, N_STEPS, features.shape[1])
    K.clear_session()
    predicted_scaled = model.predict(current_sequence)
    predicted_price = scaler_target.inverse_transform(predicted_scaled)
    print(f"Next closing price for {ticker}: {predicted_price}")

    #if now.weekday() == 0:  # Check if today is Monday
       # showNewModelPredictions(dfs, models, scalers, week_number)

# consistency_report = chesck_data_consistency(dfs)
# for key, value in consistency_report.items():
#     print(f"{key}: {value}")
