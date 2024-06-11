import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
N_STEPS = 50
LOOKUP_STEP = 1
FUTURE_STEP = [1, 2, 3, 4, 5]
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-A', 'BRK-B', 'JPM', 'JNJ', 'MFC', 'AGR', 'NEXT', 'BIGZ', 'EH']
workers = len(tickers)

# Load data
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

# Function to calculate technical indicators
def add_technical_indicators(df):
    """
    Calculate and append technical indicators to the DataFrame.
    Parameters:
    df (DataFrame): The input DataFrame containing stock prices.
    Returns:
    DataFrame: The DataFrame with added technical indicators.
    """
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Change'] = df['Close'] - df['Open']
    df['Average_Price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['RSI'] = talib.RSI(df['Close'], timeperiod=28)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)*1.5 # weight
    df['EMA_90'] = talib.EMA(df['Close'], timeperiod = 90)*1.5
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_middle'] = df['BB_middle'] * 1.75
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['Beta'] = talib.BETA(df['High'], df['Low'], timeperiod=10)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df.drop(columns=['Low', 'High', 'Stock Splits', 'BB_upper', 'BB_lower', 'Volume'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df

def add_fundamental_indicators(df, info):
    """
    Calculate and append fundamental indicators to the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing stock prices.
    info (dict): Dictionary containing fundamental information.

    Returns:
    DataFrame: The DataFrame with added fundamental indicators.
    """
    fundamental_indicators = {
        'Market_Cap': info.get('marketCap', None),
        'Enterprise_Value': info.get('enterpriseValue', None),
        'Forward_PE': info.get('forwardPE', None),
        'PEG_Ratio': info.get('pegRatio', None),
        'Beta': info.get('beta', None),
        'EBITDA': info.get('ebitda', None)
    }
    
    # Debugging: Print the info being added
    print(f"Adding fundamental indicators for {df.name}")
    for name, indicator in fundamental_indicators.items():
        print(f"{name}: {indicator}")
        df[name] = indicator

    # Ensure the DataFrame updates correctly
    df.reset_index(drop=True, inplace=True)
    
    return df

def calculate_correlation_matrix(df):
    """
    Calculates and returns the Pearson correlation matrix for the given DataFrame.
    """
    corr_matrix = df.corr()
    return corr_matrix

ticker_info = {}
for ticker in tickers:
    ticker_info[ticker] = yf.Ticker(ticker).info

# Add technical indicators to each DataFrame
for ticker, df in dfs.items():
    try:
        if not df.empty:
            df.name = ticker  # Add the ticker name for debugging purposes
            df = add_technical_indicators(df)
            #wdf = add_fundamental_indicators(df, ticker_info[ticker])
            dfs[ticker] = df
        else:
            print(f"No data available for {ticker}")
    except KeyError as e:
        print(f"Error processing {ticker}: {e}")
        continue

# Calculate and display correlation matrix for each ticker
for ticker, df in dfs.items():
    if df.empty:
        print(f"No data available for {ticker}")
        continue

    # Debugging: Verify the DataFrame contains fundamental indicators
    print(f"DataFrame for {ticker} after adding fundamental indicators:")
    print(df.tail())
    print(df.columns)
    
    corr_matrix = calculate_correlation_matrix(df)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix for {ticker}')
    plt.show()
