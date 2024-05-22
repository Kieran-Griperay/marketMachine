#Import Libraries
import yfinance as yf #Financial Analysis
import pandas as pd #DataFrames
import datetime #Timestamps
from concurrent.futures import ThreadPoolExecutor, as_completed #Parallel Processing
import talib #Technical Analysis


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
now = datetime.datetime.now()
market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0) #Get Last Day's Close
if now > market_close_time: 
    yesterday = now
else:
    yesterday = now - datetime.timedelta(days=1)
if yesterday.weekday() == 5:  # Saturday
    yesterday = yesterday - datetime.timedelta(days=1)
elif yesterday.weekday() == 6:  # Sunday
    yesterday = yesterday - datetime.timedelta(days=2)
yesterday_str = yesterday.strftime('%Y-%m-%d')
today_str = now.strftime('%Y-%m-%d')
print(f"Fetching data for {yesterday_str}.")

#Fetch Data for Single Ticker
def fetch_data(ticker):
    print(f"Processing Ticker: {ticker}") 
    try:
        stock = yf.Ticker(ticker) #Get data for Ticker
        hist = stock.history(start=(yesterday - datetime.timedelta(days=50)).strftime('%Y-%m-%d'), end=today_str) #Used for OBV
        if not hist.empty and len(hist) >= 35: #Ensure not empty date
            print(f"Data Fetched for {ticker}.")
            hist['symbol'] = ticker
            hist['marketCap'] = stock.info.get('marketCap', 'N/A') #Market Cap
            hist['sector'] = stock.info.get('sector', 'N/A') #Sector
            hist['averageVolume10days'] = stock.info.get('averageVolume10days', 'N/A') #Volume over last 10 Days
            hist['adx'] = talib.ADX(hist['High'], hist['Low'], hist['Close'], timeperiod=14)
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
yesterdays_data = yesterdays_data[['Date', 'Open', 'Close', 'Volume', 'symbol', 'marketCap', 'sector', 'averageVolume10days', 'adx']] #Relevant Columns
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
print("Top 5 Performing ETFs over the Last 3 Months: ") #Display the top 5 ETFs and their performances
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
print(filtered_by_adx.head(50)[['symbol', 'Date', 'marketCap', 'averageVolume10days','sector', 'Open', 'Close', 'ADX']])

#Filter 5: RSI (Direction)





























































































































##BACKTEST
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import talib as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# OBV Calculation Function
def calculate_obv(data):
    if 'Close' not in data.columns or 'Volume' not in data.columns:
        print("Missing 'Close' or 'Volume' columns in data.")
        return data
    data['OBV'] = ta.OBV(data['Close'], data['Volume'])
    data['OBV_Trend'] = data['OBV'].diff().apply(lambda x: 2 if x > 5 else -2 if x < -5 else 1 if x > 0 else -1 if x < 0 else 0)
    data['OBV_Trend_Sum'] = data['OBV_Trend'].rolling(window=10).sum()
    return data

# Fetch Historical Data
def fetch_historical_data(symbols, start_date, end_date):
    data = {}
    num_cores = multiprocessing.cpu_count()
    max_workers = min(2 * num_cores, 50)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(yf.Ticker(symbol).history, start=start_date, end=end_date): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data[symbol] = future.result()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
    return data

# Optimized Backtest Function
def optimized_backtest(df, start_date, end_date, lookback_period=5):
    results = []
    end_date = pd.Timestamp(end_date)
    date_range = pd.date_range(start=start_date, end=end_date - datetime.timedelta(days=lookback_period))
    symbols = df.index.unique()
    print(f"Fetching historical data for symbols: {symbols}")
    historical_data = fetch_historical_data(symbols, start_date, end_date)

    for current_date in date_range:
        current_date_str = current_date.strftime('%Y-%m-%d')
        next_date = current_date + datetime.timedelta(days=lookback_period)
        next_date_str = next_date.strftime('%Y-%m-%d')

        if next_date > end_date:
            break

        print(f"Backtesting for period: {current_date_str} to {next_date_str}")

        for symbol in symbols:
            data = historical_data.get(symbol)
            if data is not None and not data.empty:
                data_slice = data.loc[current_date_str:next_date_str]
                if len(data_slice) > 1:
                    opening_price = data_slice['Open'].iloc[0]
                    closing_price = data_slice['Close'].iloc[-1]
                    price_change = closing_price - opening_price
                    percentage_change = (price_change / opening_price) * 100

                    print(f"Symbol: {symbol}, Opening Price: {opening_price}, Closing Price: {closing_price}, Price Change: {price_change}, Percentage Change: {percentage_change}")
                    
                    stock_info = df.loc[symbol]

                    results.append({
                        'Date': current_date_str,
                        'Symbol': symbol,
                        'OpeningPrice': opening_price,
                        'ClosingPrice': closing_price,
                        'PriceChange': price_change,
                        'PercentageChange': percentage_change,
                        'AverageVolume': stock_info['averageVolume10days'],
                        'MarketSector': stock_info['sector'],
                        'MarketCap': stock_info['marketCap'],
                        'OBV': data['OBV'].iloc[-1] if 'OBV' in data.columns else pd.NA
                    })
    
    results_df = pd.DataFrame(results)
    return results_df

# Apply Dynamic Filtration and Backtest
def dynamic_filtration_and_backtest(start_date, end_date, batch_size=30):
    results = []
    end_date = pd.Timestamp(end_date)
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
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{batch_size}D')
    for current_date in date_range:
        next_date = current_date + datetime.timedelta(days=batch_size)
        if next_date > end_date:
            next_date = end_date
        current_date_str = current_date.strftime('%Y-%m-%d')
        next_date_str = next_date.strftime('%Y-%m-%d')

        filtered_data = complete_data[
            (complete_data['marketCap'] > 300000000) & 
            (complete_data['averageVolume10days'] > 500000)
        ].copy()
        filtered_data = filtered_data[filtered_data['OBV'] > 0]

        etf_data = {}
        for symbol in etf_to_sector.keys():
            etf = yf.Ticker(symbol)
            data = etf.history(start=(current_date - datetime.timedelta(days=90)).strftime('%Y-%m-%d'), end=current_date_str)
            etf_data[symbol] = data['Close']
        etf_df = pd.DataFrame(etf_data)
        returns = (etf_df.iloc[-1] / etf_df.iloc[0] - 1) * 100
        top_5_etfs = returns.sort_values(ascending=False).head(3)
        top_sectors = [etf_to_sector[etf] for etf in top_5_etfs.index if etf in etf_to_sector]
        filtered_data = filtered_data[filtered_data['sector'].isin(top_sectors)]
        filtered_data = filtered_data.sort_values(by='marketCap', ascending=False)

        basic_filtration = filtered_data.copy()
        basic_filtration.set_index('symbol', inplace=True)
        print(f"Filtering completed for date: {current_date_str}, filtered stocks: {basic_filtration.index}")
        backtest_results = optimized_backtest(basic_filtration, current_date, next_date)
        results.extend(backtest_results.to_dict('records'))
    
    results_df = pd.DataFrame(results)
    return results_df

# Define Backtest Period
start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).date()
end_date = datetime.datetime.now().date()
backtest_results = dynamic_filtration_and_backtest(start_date, end_date)
print(f"Number of results: {len(backtest_results)}")
print("Backtest Results (first 50 rows):")
print(backtest_results.head(50))

if not backtest_results.empty:
    average_return = backtest_results['PercentageChange'].mean()  # Calculate Average Returns
    print(f"Average Return: {average_return:.2f}%")
else:
    print("No valid backtest results to calculate the average return.")

# Save CSV File
results_file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtest_Results.csv'
backtest_results.to_csv(results_file_path, index=False)
print(f"Results saved to: {results_file_path}")
