#Import Libraries
import yfinance as yf #Financial Analysis
import pandas as pd #Dataframes
import numpy as np #Arrays
import talib #Technical Analysis
from sklearn.model_selection import train_test_split #Train & Test Splits
from sklearn.preprocessing import MinMaxScaler #Machine Learning
import tensorflow as tf #Machine Learning
from tensorflow import keras #Machine Learning
from tensorflow.python.keras.layers import LSTMV1, Dense, Dropout #Machine Learning
from tensorflow.python.keras.models import Sequential, load_model #Machine Learning
from tensorflow.python.keras.metrics import mean_squared_error #Machine Learning
import matplotlib.pyplot as plt #Visualizations
import sys #Python Interpreter
import joblib #Saving & Locating
import datetime #Time Manipulation
import os #File & Directory Operations

#Setup Lookback & Prediction Time Periods
LOOKUP_STEP = 1 #Predict the next day
N_STEPS = 30 #Lookback period
FUTURE_STEP = [1,2,3,4,5] #Predict next 5 days

#Determine the current week number
now = datetime.datetime.now() #Returns current data & time
week_number = now.isocalendar().week #Retrieves current week number

#Sample Stock (Create Loop For All Stock Filtered Stocks)
tickers = ['AAPL', 'NVDA', 'GOOG', 'PGR', 'SPOT', 'AMZN', 'TGT', 'TSM', 'TSLA', 'NTNX']
dfs = {}
for t in tickers:
    data = yf.Ticker(t)
    df = data.history(period='10y', interval='1d')
    dfs[t] = df

#Technial Inputs
def add_technical_indicators(df):
    technical_indicators = {
        'RSI': talib.RSI(df['Close'], timeperiod=14),
        'EMA_20': talib.SMA(df['Close'], timeperiod=20),
        'EMA_50': talib.EMA(df['Close'], timeperiod=50),
        'BB_upper': talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0],
        'BB_middle': talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1],
        'BB_lower': talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2],
        'Annual_High': df['High'].rolling(window=252).max(),
        'Annual_Low': df['Low'].rolling(window=252).min(),
        'OBV': talib.OBV(df['Close'], df['Volume'])
    }
    for name, indicator in technical_indicators.items():
        df[name] = indicator
    return df

#Fundamental Inputs
ticker_info = {}
for ticker in tickers:
    ticker_info[ticker] = yf.Ticker(ticker).info
def add_fundamental_indicators(df, info):
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

#Create LSTM datasets for each ticker
def create_dataset(X, Y, n_steps):
    Xs, Ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:(i + n_steps)])
        Ys.append(Y[i + n_steps])
    return np.array(Xs), np.array(Ys)

#Create plots for each ticker
def showNewModelPredictions(dfs, models, scalers, week_number):
    base_directory = "plots/" #Save plots to local directory
    for ticker, model in models.items(): #Loop for each ticker
        df = dfs[ticker] #Retrieve DataFrame of Ticker
        scaler_feature, scaler_target = scalers[ticker] #Retrieve Features & Targets of each Ticker
        fig_name = f"{week_number}_{ticker}" #Create unique name for each plot based on Ticker & Week
        features = scaler_feature.transform(df.drop(['Close'], axis=1))  #Scale Features
        target = scaler_target.transform(df[['Close']]) #Scale Target

        #Predictions using past Features
        predicted_close_prices = [] #Initialize list of predicted close prices for each time step
        for i in range(len(df) - N_STEPS): #Create Loop
            X_test = features[i:(i + N_STEPS)].reshape(1, N_STEPS, features.shape[1]) #Select input Features
            pred_scaled = model.predict(X_test) #Predict scaled Target
            pred_price = scaler_target.inverse_transform(pred_scaled)[0][0] #Inverse scaling
            predicted_close_prices.append(pred_price) #Appends for predicted close prices

        #Append predictions to DataFrame
        prediction_series = pd.Series(data=np.nan, index=df.index) #Initialize Prediction Series
        prediction_series[-len(predicted_close_prices):] = predicted_close_prices #Ensure close price predictions = number of elments in Series
        df['Predicted_Close'] = prediction_series #Create Column

        #Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(df['Close'], label='Actual Closing Prices')
        plt.plot(df.index[-len(predicted_close_prices):], predicted_close_prices, label='Predicted Closing Prices', linestyle='--')
        plt.title(f'Stock Price Prediction vs Validation for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.savefig(os.path.join(base_directory, f"{fig_name}_predictions.png")) #Save Plots
        plt.close() #Close Plot(Free up memory)

#Build & Train LSTM models
def build_and_train_model(data, n_steps, ticker):
    print(f"training model{ticker}")
    scaler_feature = MinMaxScaler() #Feature Object Initialized
    scaler_target = MinMaxScaler() #Target Object Initialized
    features = scaler_feature.fit_transform(data.drop(['Close'], axis=1)) #Scale Features
    target = scaler_target.fit_transform(data[['Close']].values) #Scale Target
    X, Y = create_dataset(features, target.ravel(), n_steps) #Create Inputs (X) & Outputs (Y)
    model = Sequential([
        LSTMV1(60, return_sequences=True, input_shape=(n_steps, X.shape[2])), #Creates LSTM Layer With 60 Units
        Dropout(0.3), #Adds Dropout Layer With Rate (0.3) (Prevent Overfitting)
        LSTMV1(120, return_sequences=False), #Creates LSTM Layer With 120 Units, Only Return Last Output
        Dropout(0.3), #Adds Second Dropout Layer
        Dense(20), #Dense Layers, Each unit is onnected to every unit in previous layer
        Dense(1) #Adds output layer
    ]) #Create Model
    model.compile(loss='mean_squared_error', optimizer='adam') #Mean Squared Error is the loss function
    model.fit(X, Y, epochs=50, batch_size=8, verbose=1) #Train Model with fit method
    return model, scaler_feature, scaler_target #Return Values

#Add Technical & Fundamental Indicators
data = {}
for ticker, df in dfs.items():
    info = ticker_info[ticker]
    df = add_technical_indicators(df)
    df = add_fundamental_indicators(df, info)
    data[ticker] = df

#Initialize Models & Scalers for Dictionaries
Models = {}
Scalers = {}

#Iterate Current Model for Each Ticker Each Week
for ticker, df in dfs.items(): 
    weekly_model_path = f"models/{week_number}{ticker}_model.h5" #Save for current Week & Ticker
    scaler_feature_path = f"scalers/{ticker}_scaler_feature.pkl"
    scaler_target_path = f"scalers/{ticker}_scaler_target.pkl"


    try:
        model = load_model(weekly_model_path)
        scaler_feature = joblib.load(f"scalers/{ticker}_scaler_feature.pkl")
        scaler_target = joblib.load(f"scalers/{ticker}_scaler_target.pkl")
        Models[ticker] = model #Store in 'Models' Dictionary
        Scalers[ticker] = (scaler_feature, scaler_target) #Store in 'Scalers Dictionary'
    except (IOError, FileNotFoundError):  
        print(f"Error loading model files for {ticker}")


    #Perform Train-Test Split
    X = df.drop('Close', axis=1) #Feature variables
    Y = df['Close'] #Target variables
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #Split 80/20 Train & Test Data

    #Train Set: Add indicators
    X_train = add_technical_indicators(X_train)
    X_train = add_fundamental_indicators(X_train)

    #Train Set: Build & Train 
    model, scaler_feature, scaler_target = build_and_train_model(X_train, N_STEPS, ticker)
    Models[ticker] = model
    Scalers[ticker] = (scaler_feature, scaler_target)

    #Store Scalers
    joblib.dump(scaler_feature, scaler_feature_path)
    joblib.dump(scaler_target, scaler_target_path)

    #Predict Test Set
    X_test = add_technical_indicators(X_test)
    X_test = add_fundamental_indicators(X_test)
    X_test_scaled = scaler_feature.transform(X_test)
    Y_test_scaled = scaler_target.transform(Y_test.values.reshape(-1, 1))
    X_test_processed, Y_test_processed = create_dataset(X_test_scaled, Y_test_scaled.ravel(), N_STEPS)
    Y_pred_scaled = model.predict(X_test_processed)
    Y_pred = scaler_target.inverse_transform(Y_pred_scaled)

    #Evaluate the model
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    print(f"Test RMSE for {ticker}: {test_rmse}")

    #Generate & Save Plots
    if now.weekday() == 0:  #Check if Monday
        showNewModelPredictions(dfs, Models, Scalers, week_number)