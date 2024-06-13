#Import Libraries 
import pandas as pd #DataFrames
import yfinance as yf #Financial Analysis
import talib #Technical Analysis
import matplotlib.pyplot as plt #Visualizations
import logging #Logging
import os #Directories
from multiprocessing import Pool #Processing



logging.basicConfig(level=logging.ERROR) #Logging
market_cap_stocks = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Data Collection/Market Cap Stocks.csv' #Get List of Stocks
df = pd.read_csv(market_cap_stocks) #Read list
print("Columns in the CSV file:", df.columns)
ticker_column = 'Ticker'
etf_column = 'Corresponding ETF'
sector_column = 'Sector'

if ticker_column not in df.columns: #Error Detection
    raise KeyError(f"Column '{ticker_column}' not found in the CSV file. Available columns: {df.columns}")
tickers = df[ticker_column].tolist() #Compile list
sectors = df[ticker_column].tolist() #Compile list
etfs = df[etf_column].tolist() #Compile list
unique_sectors = df['Sector'].unique() 
print("Unique Sectors in the Market Cap Stocks CSV:")
print(unique_sectors)

#Dictionary
sector_etfs = {
    'XLU': 'Utilities',
    'XLB': 'Basic Materials',
    'XLE': 'Energy',
    'XLF': 'Financial Services',
    'XLI': 'Industrials',
    'XLK': 'Technology',
    'XLP': 'Consumer Defensive',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Cyclical',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

#Backtest Periods for ETF Data
full_start_date = '2019-10-01'
full_end_date = '2023-12-31'

#Get Data to Calculate Trending ETFs
etf_data = {}
for etf in sector_etfs:
    data = yf.download(etf, start=full_start_date, end=full_end_date)
    data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)
    data['EMA_50'] = talib.EMA(data['Close'], timeperiod=50)
    data['Pct_Change'] = data['Close'].pct_change()
    data['Cumulative_Return'] = ((1 + data['Pct_Change']).cumprod() - 1)
    etf_data[etf] = data

#Filter Trending ETFs
etf_trending = {}
for etf, data in etf_data.items():
    data['EMA_20_Above_EMA_50'] = (data['EMA_20'] > data['EMA_50'])
    data['Above_EMA_20'] = (data['Close'] > data['EMA_20'])
    data['Trending_Up'] = ((data['Above_EMA_20'].rolling(window=3).sum() == 3) & data['EMA_20_Above_EMA_50'])
    etf_trending[etf] = data['Trending_Up']

#DataFrame
trending_df = pd.DataFrame(etf_trending).astype(int)
trending_df.index = pd.to_datetime(trending_df.index)
trending_file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/ETF_Trending.csv'
trending_df.to_csv(trending_file_path, index=True)
print(f"ETF Trending Information Saved at: {trending_file_path}")

#Plot Normalized Performance of Sector ETFs
plt.figure(figsize=(14, 8))
for etf, data in etf_data.items():
    plt.plot(data.index, data['Cumulative_Return'], label=etf)
plt.title('Sector ETF Performance (2020-2024)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend(loc='upper left')
plt.grid(True)
visualization_file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/ETF_Trend_Visualization.png'
plt.savefig(visualization_file_path)
print(f"ETF Trend Visualization Saved at: {visualization_file_path}")

#Directories
base_data_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting'
individual_data_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Individual Stock Data'
results_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Results'

#Get Open, Close, High, Low, Volume
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

#Technical Indicators
def calculate_indicators(data):
    try:
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14) #Average Directional Index
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14) #Relative Strength Index
        data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20) #Moving Average
        data['EMA_50'] = talib.EMA(data['Close'], timeperiod=50) #Moving Average
        data['Upper_BBand'], data['Middle_BBand'], data['Lower_BBand'] = talib.BBANDS(data['Close'], timeperiod=20) #Bollinger Bands
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9) #MACD
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14) #ATR
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14) #CCI
        data['Williams_%R'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14) #Williams %
        data['MFI'] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14) #Money Flow Index
        data['OBV'] = talib.OBV(data['Close'], data['Volume']) #On Balance Volume
        return data
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return None

#Create DataFrame
def create_dataframe(ticker, etf, sector, start_date, final_end_date):
    print(f"Processing {ticker} with ETF {etf} in sector {sector}")
    trending_file_path = os.path.join(base_data_dir, 'ETF_Trending.csv')
    trending_df = pd.read_csv(trending_file_path, index_col=0, parse_dates=True) #Get ETF Trending Sector Data
    if etf not in trending_df.columns: #Error Detection
        print(f"No matching ETF column found for {etf}")
        return None
    trending_dates = trending_df[etf] 
    if not trending_dates.any(): #Error Detection
        print(f"No trending dates for ETF {etf}")
        return None
    data = fetch_stock_data(ticker, start_date, final_end_date) #Get Data for Specified Time Period
    if data is None:
        return None
    data = calculate_indicators(data) #Make Calculations on Data
    if data is None:
        return None
    df_combined = data.copy() #Avoid Slicing Warning
    df_combined['Ticker'] = ticker 
    df_combined['Corresponding ETF'] = etf 
    df_combined['Sector'] = sector
    df_combined['SP'] = trending_dates.reindex(data.index, fill_value=0)
    df_combined['Market Cap'] = df[df[ticker_column] == ticker]['Market Cap'].values[0]
    df_combined = df_combined.dropna(subset=['ADX', 'RSI', 'EMA_20', 'EMA_50', 'Upper_BBand', 'Middle_BBand', #Drop NaN Rows
                                             'Lower_BBand', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'CCI', 'Williams_%R',
                                             'MFI', 'OBV', 'SP'])
    return df_combined

#Process Single Ticker
def process_ticker(ticker_info):
    selected_ticker, df, ticker_column, etf_column, sector_column, lookback_start_date, final_end_date = ticker_info
    try:
        selected_etf = df[df[ticker_column] == selected_ticker][etf_column].values[0] #Assign ETF to each Ticker
        selected_sector = df[df[ticker_column] == selected_ticker][sector_column].values[0] #Assign Sector to each Ticker
        stock_df = create_dataframe(selected_ticker, selected_etf, selected_sector, lookback_start_date, final_end_date) #Call Create DataFrame Method
        if stock_df is not None: #Error Detection
            stock_df_path = f'/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Individual Stock Data/{selected_ticker}_Data.csv' #Save to Folder
            stock_df.to_csv(stock_df_path, index=True) #Read CSV
            print(f"Data for {selected_ticker} saved at {stock_df_path}")
            return stock_df_path
        else: #Error Detection
            print(f"Failed to process {selected_ticker}")
            return None 
    except Exception as e: #Error Detection
        logging.error(f"Error processing ticker {selected_ticker}: {e}")
        return None

#Main Function
def main():
    ticker_column = 'Ticker'
    etf_column = 'Corresponding ETF'
    sector_column = 'Sector'
    lookback_start_date = pd.to_datetime('2019-10-01') 
    final_end_date = pd.to_datetime('2023-12-31')
    tickers = df[ticker_column].unique()
    ticker_info_list = [(ticker, df, ticker_column, etf_column, sector_column, lookback_start_date, final_end_date) for ticker in tickers] #Create List of Tuples
    with Pool(processes=os.cpu_count()) as pool: #Multi Processing
        pool.map(process_ticker, ticker_info_list)

#Run
if __name__ == "__main__": 
    main()
