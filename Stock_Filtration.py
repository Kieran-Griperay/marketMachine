#Import Libraries
import yfinance as yf #Financial Analysis
import pandas as pd #DataFrames
import datetime #Timestamps
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed #Parallel Processing
import talib #Technical Analysis
from ta.momentum import RSIIndicator #Backtest
from ta.trend import ADXIndicator #Backtest
import multiprocessing #Backtest



##DATA COLLECTION
#Original CSV
print("Loading CSV File...") 
file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Merged_Stocks.csv' #Get list of stocks
tickers_df = pd.read_csv(file_path) #Read stocks
print("CSV file loaded successfully.")
tickers = tickers_df['symbol'].tolist() #Extract Tickers using 'symbol' column
print(f"Extracted {len(tickers)} Tickers.") #How many Tickers are there

#New DataFrame
yesterdays_data = pd.DataFrame()

#Timing
now = datetime.now()
market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0) #Get Last Day's Close
if now > market_close_time: 
    yesterday = now
else:
    yesterday = now - timedelta(days=1)
if yesterday.weekday() == 5:  # Saturday
    yesterday = yesterday - timedelta(days=1)
elif yesterday.weekday() == 6:  # Sunday
    yesterday = yesterday - timedelta(days=2)
yesterday_str = yesterday.strftime('%Y-%m-%d')
today_str = now.strftime('%Y-%m-%d')
print(f"Fetching data for {yesterday_str}.")

#Fetch Data for Single Ticker
def fetch_data(ticker):
    print(f"Processing Ticker: {ticker}") 
    try:
        stock = yf.Ticker(ticker) #Get data for Ticker
        hist = stock.history(start=(yesterday - timedelta(days=50)).strftime('%Y-%m-%d'), end=today_str) #Used for OBV
        if not hist.empty and len(hist) >= 35: #Ensure not empty date
            print(f"Data Fetched for {ticker}.")
            hist['symbol'] = ticker
            hist['marketCap'] = stock.info.get('marketCap', 'N/A') #Market Cap
            hist['sector'] = stock.info.get('sector', 'N/A') #Sector
            hist['averageVolume10days'] = stock.info.get('averageVolume10days', 'N/A') #Volume over last 10 Days
            hist['ADX'] = talib.ADX(hist['High'], hist['Low'], hist['Close'], timeperiod=14)
            hist['RSI'] = talib.RSI(hist['Close'], timeperiod=14)
            #Keep Last Day's Data
            last_day_data = hist.iloc[-1:]
            return last_day_data
        else:
            print(f"No data available for {ticker}.") #Error Detection
            return None
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}") #Error Detection
        return None
    
#Parallel Processing
with ThreadPoolExecutor(max_workers=50) as executor: #50 workers
    futures = {executor.submit(fetch_data, ticker): ticker for ticker in tickers} #Dictionary to map futures to tickers
    for future in as_completed(futures): #Iterates
        result = future.result() #Store results
        if result is not None:
            yesterdays_data = pd.concat([yesterdays_data, result]) #Join Data

#Clean up DataFrame
yesterdays_data.reset_index(inplace=True)
print("Columns in DataFrame:", yesterdays_data.columns) #Print columns
yesterdays_data['marketCap'] = pd.to_numeric(yesterdays_data['marketCap'], errors='coerce') #Numeric
yesterdays_data['averageVolume10days'] = pd.to_numeric(yesterdays_data['averageVolume10days'], errors='coerce') #Numeric
print("Data types in DataFrame before conversion:")
print(yesterdays_data.dtypes)
yesterdays_data = yesterdays_data[['Date', 'Open', 'Close', 'Volume', 'symbol', 'marketCap', 'sector', 'averageVolume10days', 'ADX', 'RSI']] #Relevant Columns
output_file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Yesterdays_Stock_Data.csv' #File Path
yesterdays_data.to_csv(output_file_path, index=False) #Save to Csv
print(f"Data Fetched & Saved to: '{output_file_path}'")




##Filtration (Size, Liquidity, Sector, Momentum, Direction?)
#Total Stocks
print(f"Number of Starting Stocks: {len(yesterdays_data)}")

#FILTER 1: Market Cap (Size)
filtered_by_market_cap = yesterdays_data[yesterdays_data['marketCap'] > 300000000]
print(f"Number of Stocks After Market Cap Filter: {len(filtered_by_market_cap)}")

#FILTER 2: Average Volume (Liquidity)
filtered_by_volume = filtered_by_market_cap[filtered_by_market_cap['averageVolume10days'] > 500000].copy()
print(f"Number of Stocks After Volume Filter: {len(filtered_by_volume)}")

#FILTER 3: Trending ETF (Sector)
etf_symbols = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLU', 'XLC']
etf_data = {} #Initialize
for symbol in etf_symbols: #Track ETF Performances
    etf = yf.Ticker(symbol) 
    data = etf.history(period="3mo") #Fetch data for last 3 months
    etf_data[symbol] = data['Close']  
etf_df = pd.DataFrame(etf_data) #Save into DataFrame
returns = (etf_df.iloc[-1] / etf_df.iloc[0] - 1) * 100  #Calculate Return as a Percentage
top_3_etfs = returns.sort_values(ascending=False).head(3) #Sort the returns to find the top 3 performing ETFs
print("Top 3 Performing ETFs over the Last 3 Months: ") #Display the top 5 ETFs and their performances
print(top_3_etfs.head()) 
#Map ETFs to their sectors
etf_to_sector = {
    'XLY': 'Consumer Cyclical',
    'XLP': 'Consumer Defensive',
    'XLE': 'Energy',
    'XLF': 'Financial Services',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLK': 'Technology',
    'XLB': 'Basic Materials',
    'XLRE': 'Real Estate',
    'XLU': 'Utilities',
    'XLC': 'Communication Services'
}
top_sectors = [etf_to_sector[etf] for etf in top_3_etfs.index if etf in etf_to_sector] #Identify Top 3 Sectors
filtered_by_sector = filtered_by_volume[filtered_by_volume['sector'].isin(top_sectors)] #Filter by sector
filtered_by_sector = filtered_by_sector.reset_index()
filtered_by_sector = filtered_by_sector.sort_values(by='marketCap', ascending=False) #Sort by Market Cap
print(f"Number of Stocks After Sector Filter: {len(filtered_by_sector)}")

#FILTER 4: ADX (Momentum)
filtered_by_adx = filtered_by_sector[filtered_by_sector['ADX'] > 35] #High Momentum
print(f"Number of Stocks After ADX Filter: {len(filtered_by_adx)}")

#Filter 5: RSI (Direction)
filtered_by_rsi = filtered_by_adx[filtered_by_adx['RSI'] > 50] #Upwards Direction
print(f"Number of Stocks After RSI Filter: {len(filtered_by_rsi)}")
print(filtered_by_rsi.head(50)[['symbol', 'Date', 'marketCap', 'averageVolume10days', 'sector', 'Open', 'Close', 'ADX', 'RSI']])






##Backtest
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from concurrent.futures import ThreadPoolExecutor

# Define the function to fetch historical data
def fetch_data_in_batches(tickers, start_date, end_date, batch_size=50):
    data = []
    tickers_list = tickers.tolist()
    for i in range(0, len(tickers_list), batch_size):
        batch = tickers_list[i:i + batch_size]
        batch_data = yf.download(batch, start=start_date, end=end_date)
        data.append(batch_data)
    return pd.concat(data)

# Calculate RSI
def calculate_rsi(data, window=14):
    rsi = RSIIndicator(data['Close'], window=window)
    data['RSI'] = rsi.rsi()
    return data

# Calculate ADX
def calculate_adx(data, window=14):
    adx = ADXIndicator(data['High'], data['Low'], data['Close'], window=window)
    data['ADX'] = adx.adx()
    return data

# Calculate average volume
def calculate_avg_volume(data, window=10):
    data['averageVolume10days'] = data['Volume'].rolling(window=window).mean()
    return data

# Calculate returns for holding period
def calculate_returns(start_price, end_price):
    return (end_price - start_price) / start_price

# Define the time periods for the backtest
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

# Fetch ETF data for the previous 3 months
etf_start_date = start_date - timedelta(days=90)
etf_symbols = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLU', 'XLC']
etf_data = {symbol: yf.Ticker(symbol).history(start=etf_start_date, end=start_date)['Close'] for symbol in etf_symbols}
etf_df = pd.DataFrame(etf_data)
etf_returns = (etf_df.iloc[-1] / etf_df.iloc[0] - 1) * 100
top_3_etfs = etf_returns.sort_values(ascending=False).head(3)
etf_to_sector = {
    'XLY': 'Consumer Cyclical', 'XLP': 'Consumer Defensive', 'XLE': 'Energy',
    'XLF': 'Financial Services', 'XLV': 'Healthcare', 'XLI': 'Industrials',
    'XLK': 'Technology', 'XLB': 'Basic Materials', 'XLRE': 'Real Estate',
    'XLU': 'Utilities', 'XLC': 'Communication Services'
}
top_sectors = [etf_to_sector[etf] for etf in top_3_etfs.index if etf in etf_to_sector]

# Fetch historical data for the entire stock universe
all_tickers = yesterdays_data['symbol'].unique()
historical_data = fetch_data(all_tickers, start_date, end_date)

# Initialize an empty dataframe to store results
results = pd.DataFrame(columns=['Date', 'Symbol', 'Open_Price', 'Close_Price_5d', 'Return',
                                'RSI', 'ADX', 'Sector', 'MarketCap', 'AverageVolume'])

# Define the holding period
holding_period = 5  # Number of days stock is held

# Define the function to process each date
def process_date(date):
    date_results = []

    # Calculate indicators dynamically
    date_data = historical_data.loc[:date].copy()
    if date_data.empty:
        return date_results

    date_data = date_data.groupby('symbol').apply(calculate_rsi)
    date_data = date_data.groupby('symbol').apply(calculate_adx)
    date_data = date_data.groupby('symbol').apply(calculate_avg_volume)

    # Get previous day's data
    yesterday = date - timedelta(days=1)
    yesterday_data = date_data[date_data.index.get_level_values(1) == yesterday]

    if yesterday_data.empty:
        return date_results

    # Dynamically fetch market cap and sector for each stock
    market_cap_sector = {}
    for symbol in yesterday_data['symbol'].unique():
        ticker = yf.Ticker(symbol)
        info = ticker.info
        market_cap_sector[symbol] = {
            'marketCap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown')
        }
    yesterday_data['marketCap'] = yesterday_data['symbol'].map(lambda x: market_cap_sector[x]['marketCap'])
    yesterday_data['sector'] = yesterday_data['symbol'].map(lambda x: market_cap_sector[x]['sector'])

    # Apply filters
    filtered_by_market_cap = yesterday_data[yesterday_data['marketCap'] > 300000000]
    filtered_by_volume = filtered_by_market_cap[filtered_by_market_cap['averageVolume10days'] > 500000]
    filtered_by_sector = filtered_by_volume[filtered_by_volume['sector'].isin(top_sectors)]

    # Filter by ADX and RSI
    filtered_by_adx = filtered_by_sector[filtered_by_sector['ADX'] > 35]
    filtered_by_rsi = filtered_by_adx[filtered_by_adx['RSI'] > 50]

    for _, row in filtered_by_rsi.iterrows():
        symbol = row['symbol']

        try:
            # Get the open price on the current date and the close price 5 days later
            open_price = historical_data.loc[(symbol, date), 'Open']
            end_date_holding = date + timedelta(days=holding_period)

            if end_date_holding not in historical_data.index.get_level_values(1):
                continue

            close_price_5d = historical_data.loc[(symbol, end_date_holding), 'Close']
            stock_return = calculate_returns(open_price, close_price_5d)

            date_results.append({
                'Date': date,
                'Symbol': symbol,
                'Open_Price': open_price,
                'Close_Price_5d': close_price_5d,
                'Return': stock_return,
                'RSI': row['RSI'],
                'ADX': row['ADX'],
                'Sector': row['sector'],
                'MarketCap': row['marketCap'],
                'AverageVolume': row['averageVolume10days']
            })

        except KeyError as e:
            print(f"KeyError for {symbol} on {date}: {e}")
        except Exception as e:
            print(f"Error processing {symbol} on {date}: {e}")

    return date_results

# Use ThreadPoolExecutor to parallelize the processing
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_date, date) for date in pd.date_range(start_date, end_date - timedelta(days=holding_period), freq='B')]
    for future in as_completed(futures):
        results = results.append(future.result(), ignore_index=True)

# Analyze the results
average_return = results['Return'].mean()
total_return = (1 + results['Return']).prod() - 1

print(f"Average Return: {average_return * 100:.2f}%")
print(f"Total Return: {total_return * 100:.2f}%")
print(results.head(50))

# Save results to CSV for further analysis
results.to_csv('/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtest_Results.csv', index=False)
