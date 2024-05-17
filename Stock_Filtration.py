#Import Libraries
import pandas as pd #DataFrames
import yfinance as yf #Financial Analysis
import time #Timestamps
from concurrent.futures import ThreadPoolExecutor, as_completed #Parallel Processing
##Chabnge

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
                   'profitMargins', 'sector',]
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
    with ThreadPoolExecutor(max_workers=20) as executor: #Process 20 simultaneously
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


##FILTER 1: Market Cap 
filtered_by_market_cap = complete_data[complete_data['marketCap'] > 300000000]
print(f"Number of Stocks After Market Cap Filter: {len(filtered_by_market_cap)}")


##FILTER 2: Average Volume
filtered_by_volume = filtered_by_market_cap[filtered_by_market_cap['averageVolume10days'] > 100000].copy()
print(f"Number of Stocks After Volume Filter: {len(filtered_by_volume)}")


##FILTER 3: Relative Volume 
filtered_by_volume['relativeVolume'] = filtered_by_volume['volume'] / filtered_by_volume['averageVolume10days']
filtered_by_relative_volume = filtered_by_volume[filtered_by_volume['relativeVolume'] > 2]
print(f"Number of Stocks After Relative Volume Filter: {len(filtered_by_relative_volume)}")


##FILTER 4: Trending Sector
etf_symbols = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLU', 'XLC']
etf_data = {} #Initialize
for symbol in etf_symbols: #Track ETF Performances
    etf = yf.Ticker(symbol) 
    data = etf.history(period="3mo") #Fetch data for last 3 months
    etf_data[symbol] = data['Close']  
etf_df = pd.DataFrame(etf_data) #Save into DataFrame
returns = (etf_df.iloc[-1] / etf_df.iloc[0] - 1) * 100  #Calculate Return as a Percentage
top_5_etfs = returns.sort_values(ascending=False).head(5) #Sort the returns to find the top 5 performing ETFs
print(f"Top 5 Performing ETFs over the Last 3 Months:", top_5_etfs) #Display the top 5 ETFs and their performances

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
top_sectors = [etf_to_sector[etf] for etf in top_5_etfs if etf in etf_to_sector] #Identify Top 5 Sectors
filtered_by_sector = filtered_by_relative_volume[filtered_by_relative_volume['sector'].isin(top_sectors)] #Filter by sector
print(f"Number of Stocks After Sector Filter: {len(filtered_by_sector)}")

