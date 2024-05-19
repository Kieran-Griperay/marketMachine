#Import Libraries
import pandas as pd #DataFrames
import yfinance as yf #Financial Analysis
import time #Timestamps
from concurrent.futures import ThreadPoolExecutor, as_completed #Parallel Processing
import datetime #Backtesting



##DATA COLLECTION
#Get Stocks from NASDAQ & NYSE
file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Data Collection/Stock_Listings.csv' #Store file path
stock_data = pd.read_csv(file_path) #Read file path
filtered_stocks = stock_data[(stock_data['exchange'] == 'NASDAQ') | (stock_data['exchange'] == 'NYSE')] #Select Stocks on NASDAQ or NYSE
filtered_stocks.to_csv('/Users/ryangalitzdorfer/Downloads/Market Machine/Data Collection/NYSE_&_NASDAQ_Stocks.csv', index=False) #Save as new CSV
print(filtered_stocks.head(50)) #See DataFrame
print(len(filtered_stocks)) #See number of stocks

#Mapping for exchanges to their Yahoo Finance codes
exchange_mapping = {
    'NASDAQ': '',  #Empty string for NASDAQ 
    'NYSE': ''     #Empty string for NYSE 
}

#Initialize Dictionary with useful metrics for filtration
additional_info = ['marketCap', 'trailingPE', 'forwardPE', 'dividendYield', 'volume' ,'averageVolume10days',
                   'profitMargins', 'sector', 'Open', 'Close']
extended_data = {info: [] for info in ['symbol'] + additional_info} #Store fetched data for each symbol

#Preferred stock, warrant, or specific index
def is_preferred_or_warrant(ticker):
    return '-' in ticker or '.' in ticker

#Process a single ticker and fetch data
def fetch_data_for_ticker(row):
    ticker = row['symbol']
    exchange = row['exchange']
    if exchange in exchange_mapping: #Adjust ticker using exchange mapping
        ticker = f"{ticker}"  
    if is_preferred_or_warrant(ticker): #Skip preferred stocks and warrants
        #print(f"Skipping preferred stock or warrant: {ticker}") #Error Detection
        return None
    stock = yf.Ticker(ticker) #Creates instance
    try:
        info = stock.info #Fetches financial data
        if info and 'marketCap' in info: #Check if Yahoo can find data
            data = {'symbol': row['symbol']} #Store individual stock info
            for field in additional_info: #Retrieve corresponding data from Yahoo
                data[field] = info.get(field, pd.NA) #Store data
            return data
        else:
            print("") #Error Detection
            #print(f"No data found for {ticker}, skipping...") #Error detection
    except Exception as e:
        print("") #Error Detection
        print(f"Error fetching data for {ticker}: {e}") #Error detection
    return None

#Parallel Processing
def process_stocks_in_parallel(stocks):
    results = []
    with ThreadPoolExecutor(max_workers=50) as executor: #Process 50 simultaneously
        futures = [executor.submit(fetch_data_for_ticker, row) for _, row in stocks.iterrows()] #Iterates through each row
        for future in as_completed(futures): #Process results instantly
            result = future.result() #Store results
            if result:
                results.append(result) #Add results to list
    return results

results = process_stocks_in_parallel(filtered_stocks) #Apply function
for result in results: #Update extended data
    for key, value in result.items():
        extended_data[key].append(value)

additional_df = pd.DataFrame(extended_data) #Convert to DataFrame

successful_stocks = len(extended_data['symbol']) #Get number of stocks succesfully processed
print(f"Number of Successfully Processed Stocks: {successful_stocks}") #Print length of dataframe

complete_data = pd.merge(filtered_stocks, additional_df, on='symbol', how='inner') #Merge both Datasets into Complete Dataset
complete_data.to_csv('/Users/ryangalitzdorfer/Downloads/Market Machine/Data Collection/Complete_Stocks.csv', index=False) #Save 'Complete Stocks' to DataFrame 



##Basic Filtration (Size, Liquidity, Attention, Sector)
#FILTER 1: Market Cap 
filtered_by_market_cap = complete_data[complete_data['marketCap'] > 300000000]
print(f"Number of Stocks After Market Cap Filter: {len(filtered_by_market_cap)}")

#FILTER 2: Average Volume
filtered_by_volume = filtered_by_market_cap[filtered_by_market_cap['averageVolume10days'] > 500000].copy()
print(f"Number of Stocks After Volume Filter: {len(filtered_by_volume)}")

#FILTER 3: Relative Volume 
filtered_by_volume['relativeVolume'] = filtered_by_volume['volume'] / filtered_by_volume['averageVolume10days']
filtered_by_relative_volume = filtered_by_volume[filtered_by_volume['relativeVolume'] > 2]
print(f"Number of Stocks After Relative Volume Filter: {len(filtered_by_relative_volume)}")

#FILTER 4: Trending Sector
etf_symbols = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLU', 'XLC']
etf_data = {} #Initialize
for symbol in etf_symbols: #Track ETF Performances
    etf = yf.Ticker(symbol) 
    data = etf.history(period="3mo") #Fetch data for last 3 months
    etf_data[symbol] = data['Close']  
etf_df = pd.DataFrame(etf_data) #Save into DataFrame
returns = (etf_df.iloc[-1] / etf_df.iloc[0] - 1) * 100  #Calculate Return as a Percentage
top_5_etfs = returns.sort_values(ascending=False).head(5) #Sort the returns to find the top 5 performing ETFs
print("Top 5 Performing ETFs over the Last 3 Months: ") #Display the top 5 ETFs and their performances
print(top_5_etfs.head()) 
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
top_sectors = [etf_to_sector[etf] for etf in top_5_etfs.index if etf in etf_to_sector] #Identify Top 5 Sectors
filtered_by_sector = filtered_by_relative_volume[filtered_by_relative_volume['sector'].isin(top_sectors)] #Filter by sector
filtered_by_sector = filtered_by_sector.sort_values(by='marketCap', ascending=False) #Sort by Market Cap
print(f"Number of Stocks After Sector Filter: {len(filtered_by_sector)}")
print(f"Today's Selected Stocks", filtered_by_sector.head(50)) #See Stock Selection



##BACKTEST
#Get Historical Data
def fetch_historical_data(symbols, start_date, end_date):
    data = {}
    with ThreadPoolExecutor(max_workers=50) as executor:  #Increased max workers
        futures = {executor.submit(yf.Ticker(symbol).history, start=start_date, end=end_date): symbol for symbol in symbols} #Fetch for each symbol
        for future in as_completed(futures): #Iterate over future objects
            symbol = futures[future] #Retrieve symbol from futures dictionary
            try: #Error detection
                data[symbol] = future.result() #Historical data stored in dictionary
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
    return data

#Optimized backtest function
def optimized_backtest(df, start_date, end_date, lookback_period=5):
    results = []
    end_date = pd.Timestamp(end_date) #Timestamp for comparison
    date_range = pd.date_range(start=start_date, end=end_date - datetime.timedelta(days=lookback_period)) 
    symbols = df.index.unique() #Get unique symbols
    print(f"Fetching historical data for symbols: {symbols}")
    historical_data = fetch_historical_data(symbols, start_date, end_date) #Use historical data function
    #Iterate over each date from lookback period
    for current_date in date_range:
        current_date_str = current_date.strftime('%Y-%m-%d')
        next_date = current_date + datetime.timedelta(days=lookback_period)
        next_date_str = next_date.strftime('%Y-%m-%d')
        #Continue through loop
        if next_date > end_date:
            break #Error detection
        print(f"Backtesting for period: {current_date_str} to {next_date_str}")
        #Go through each symbol for each day
        for symbol in symbols:
            data = historical_data.get(symbol) #Get data
            if data is not None:
                data_slice = data.loc[current_date_str:next_date_str]
                if len(data_slice) > 1: #Ensure we have enough data points
                    opening_price = data_slice['Open'].iloc[0] #Get opening price
                    closing_price = data_slice['Close'].iloc[-1] #Get closing price
                    price_change = closing_price - opening_price #Get price change
                    percentage_change = (price_change / opening_price) * 100 #Get percentage change
                    print(f"Symbol: {symbol}, Opening Price: {opening_price}, Closing Price: {closing_price}, Price Change: {price_change}, Percentage Change: {percentage_change}") 
                    #Store dictionary
                    results.append({
                        'Date': current_date_str,
                        'Symbol': symbol,
                        'OpeningPrice': opening_price,
                        'ClosingPrice': closing_price,
                        'PriceChange': price_change,
                        'PercentageChange': percentage_change
                    })
    #Store in DataFrame                
    results_df = pd.DataFrame(results)
    return results_df

#Apply Dynamic Filtration 
def dynamic_filtration_and_backtest(start_date, end_date, batch_size=30):
    results = []
    end_date = pd.Timestamp(end_date) 
    #Map ETFs
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
    #Iterate over specified dates
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{batch_size}D') #Set range
    for current_date in date_range:
        next_date = current_date + datetime.timedelta(days=batch_size)
        if next_date > end_date:
            next_date = end_date
        current_date_str = current_date.strftime('%Y-%m-%d')
        next_date_str = next_date.strftime('%Y-%m-%d')
        #Apply Filtration methods
        filtered_data = complete_data[
            (complete_data['marketCap'] > 300000000) & #Market Cap
            (complete_data['averageVolume10days'] > 500000) #Average Volume
        ].copy()
        filtered_data['relativeVolume'] = filtered_data['volume'] / filtered_data['averageVolume10days']
        filtered_data = filtered_data[filtered_data['relativeVolume'] > 2] #Relative Volume
        #Trending Sectors
        etf_data = {}
        for symbol in etf_to_sector.keys():
            etf = yf.Ticker(symbol)
            data = etf.history(start=(current_date - datetime.timedelta(days=90)).strftime('%Y-%m-%d'), end=current_date_str)
            etf_data[symbol] = data['Close']
        etf_df = pd.DataFrame(etf_data)
        returns = (etf_df.iloc[-1] / etf_df.iloc[0] - 1) * 100 #Calculate percentage
        top_5_etfs = returns.sort_values(ascending=False).head(5)
        top_sectors = [etf_to_sector[etf] for etf in top_5_etfs.index if etf in etf_to_sector]
        filtered_data = filtered_data[filtered_data['sector'].isin(top_sectors)]
        filtered_data = filtered_data.sort_values(by='marketCap', ascending=False) #Sort by Market Cap
        #Create DataFrame to store results
        basic_filtration = filtered_data.copy()
        basic_filtration.set_index('symbol', inplace=True)
        print(f"Filtering completed for date: {current_date_str}, filtered stocks: {basic_filtration.index}")
        backtest_results = optimized_backtest(basic_filtration, current_date, next_date) 
        results.extend(backtest_results.to_dict('records'))
    results_df = pd.DataFrame(results) 
    return results_df
#Define Backtest Period
start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).date()
end_date = datetime.datetime.now().date()
backtest_results = dynamic_filtration_and_backtest(start_date, end_date)
print(f"Number of results: {len(backtest_results)}")
print("Backtest Results (first 50 rows):")
print(backtest_results.head(50))
average_return = backtest_results['PercentageChange'].mean() #Calculate Average Returns
print(f"Average Return: {average_return:.2f}%")
#Save CSV File
results_file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Data Collection/Backtest_Results.csv'
backtest_results.to_csv(results_file_path, index=False)
print(f"Results saved to: {results_file_path}")

