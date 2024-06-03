#Import Libraries 
import pandas as pd #DataFrames
import yfinance as yf #Finance 
from concurrent.futures import ThreadPoolExecutor, as_completed #Parallel Processing
import logging #Logging

#Log
logging.basicConfig(filename='/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Data Collection/Market_Cap_Errors.log', level=logging.ERROR, 
                    format='%(asctime)s:%Y-%m-%d %H:%M:%S:%message)s') #Error Detection
#Read list of all NASDAQ & NYSE stocks
file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Data Collection/Merged_Stocks.csv'
df = pd.read_csv(file_path)
print("Columns in the CSV file:", df.columns) #Error Detection
ticker_column = 'symbol' #Symbol
if ticker_column not in df.columns: #Error Detection
    raise KeyError(f"Column '{ticker_column}' not found in the CSV file. Available columns: {df.columns}")

#Mapping for Column 'Corresponding ETF'
sector_etfs = {
    'Utilities': 'XLU',
    'Basic Materials': 'XLB',
    'Energy': 'XLE',
    'Financial Services': 'XLF',
    'Industrials': 'XLI',
    'Technology': 'XLK',
    'Consumer Defensive': 'XLP',
    'Healthcare': 'XLV',
    'Consumer Cyclical': 'XLY',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    'Unknown': 'Unknown'  #Handle Unknown Sectors
}

#Filter by Market Cap
def get_sector_market_cap_etf(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        market_cap = info.get('marketCap', None)
        etf = sector_etfs.get(sector, 'Unknown')
        return ticker, sector, market_cap, etf
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return ticker, 'Unknown', None, 'Unknown'
tickers = df[ticker_column].tolist() #Send Tickers to List
stocks_data =[]

#Parallel Processing
with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(get_sector_market_cap_etf, ticker) for ticker in tickers]
    for future in as_completed(futures):
        ticker, sector, market_cap, etf = future.result()
        if market_cap is not None and market_cap >= 300e6: # Small Cap or above
            stocks_data.append((ticker, sector, market_cap, etf))

#DataFrame from the stocks data
stocks_df = pd.DataFrame(stocks_data, columns=['Ticker', 'Sector', 'Market Cap', 'Corresponding ETF'])
output_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Data Collection/Market Cap Stocks.csv'
stocks_df.to_csv(output_path, index=False)
print(f"Small-Cap + Tickers Saved to {output_path}")