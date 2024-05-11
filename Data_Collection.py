#Import Libraries
import yfinance as yf #Financial Analysis
import pandas as pd #Dataframes
import numpy as np #Arrays
import talib #Technical Analysis
from sklearn.model_selection import train_test_split #Train & Test Splits
from sklearn.preprocessing import MinMaxScaler #Machine Learning
import tensorflow as tf #Machine Learning
from tensorflow import keras #Machine Learning
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt #Visualizations
import pkg_resources

# Check TensorFlow installation
try:
    print("TensorFlow version:", tf.__version__)
except ImportError:
    print("TensorFlow is not installed.")

# Check for conflicting dependencies
print("\nInstalled packages:")
for package in pkg_resources.working_set:
    print(package)

# Verify Python environment
import sys
print("\nPython executable path:")
print(sys.executable)


#Setup Lookback & Prediction Time Periods
LOOKUP_STEP = 1 #Predict the next day
for days in range(10, 51, 10):
    N_STEPS = days // LOOKUP_STEP #:Lookback period
    print(f"N_STEPS for {days} days: {N_STEPS}") 
FUTURE_STEP = [1,2,3,4,5] #Predict next 5 days

#Sample Stock (Create Loop For All Stock Filtered Stocks)
ticker = 'AAPL'
data = yf.Ticker(ticker)
df= data.history(period='10y', interval='1d')

#Technial Inputs
df['RSI'] = talib.RSI(df['Close'], timeperiod=14) #Relative Strength Index
df['EMA_20'] = talib.SMA(df['Close'], timeperiod=50) #20 Period Exponential Moving Average
df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50) #50 Period Exponential Moving Average
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9) #Moving Average Convergence Divergence
df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0) #Bollinger Bands
df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14) #Average True Range
df['Annual_High'] = df['High'].rolling(window=252).max() #Annual High
df['Annual_Low'] = df['Low'].rolling(window=252).min() #Annual Low
df['OBV'] = talib.OBV(df['Close'], df['Volume']) #On Balance Volume

#Fundamental Inputs
info = data.info
df['Market_Cap'] = info.get('marketCap', None) 
df['Enterprise_Value'] = info.get('enterpriseValue', None)
df['Forward_PE'] = info.get('forwardPE', None)
df['Trailing_PE'] = info.get('trailingPE', None)
df['PEG_Ratio'] = info.get('pegRatio', None)
df['Book_Value'] = info.get('bookValue', None)
df['Dividend_Rate'] = info.get('dividendRate', None)
df['Dividend_Yield'] = info.get('dividendYield', None)

#Drop Null Values
df.dropna(inplace=True)

#Scale Features & Targets
scaler_feature = MinMaxScaler()
scaled_features = scaler_feature.fit_transform(df.drop(['Close'], axis=1))
scaler_target = MinMaxScaler()
scaled_target = scaler_target.fit_transform(df[['Close']])

#Data Preparation
NUM_features = scaled_features.shape[1] #Number of Features
Total_elements = scaled_features.shape[0] * N_STEPS * NUM_features #Calculate Desired Array Shape
if scaled_features.size > Total_elements: #Padding & Truncating for Reshaping
    scaled_features = scaled_features[:Total_elements]
elif scaled_features.size < Total_elements:
    num_elements_to_pad = Total_elements - scaled_features.size
    scaled_features = np.pad(scaled_features, ((0, num_elements_to_pad)), mode='constant', constant_values=0)
X = scaled_features.reshape(-1, N_STEPS, NUM_features) #Define X
Y = scaled_target #Define Y

#Train & Test Splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
Total_elements_train = X_train.shape[0] * N_STEPS * NUM_features #Desired Array for Features in Training Set

#TensorFlow Model
model = Sequential([
])
model.compile(optimizer='adam', loss='mean_squared_error') #Compile Model 
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2) #Model Performance Over History
loss = model.evaluate(X_test, Y_test) #Evaluate Test Data With Mean Squared Error

# Scale Features & Targets for Test Set
scaled_features_test = scaler_feature.transform(X_test.reshape(-1, N_STEPS, NUM_features))
scaled_target_test = scaler_target.transform(Y_test)

# Reshape Features for Model Input
X_test_reshaped = scaled_features_test.reshape(-1, N_STEPS, NUM_features)

# Evaluate Test Data With Mean Squared Error
loss_test = model.evaluate(X_test_reshaped, scaled_target_test)

print("Test Loss (Mean Squared Error):", loss_test)







print(df.columns)