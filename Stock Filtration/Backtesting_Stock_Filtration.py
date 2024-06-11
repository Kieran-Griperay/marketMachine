#Import Libraries
import pandas as pd #DataFrame Manipulation
import yfinance as yf #Descriptive Analysis
import talib #Technical Analysis
import matplotlib.pyplot as plt #Visualizations
from concurrent.futures import ThreadPoolExecutor, as_completed #Parallel Processing
import logging #Logging
import os #Operating System
from multiprocessing import Pool #Processing


#List of All Stocks With Market Cap Above 300 Million
market_cap_stocks = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Data Collection/Market Cap Stocks.csv' ##Filter 1
df = pd.read_csv(market_cap_stocks)
print("Columns in the CSV file:", df.columns)
ticker_column = 'Ticker' 
etf_column = 'Corresponding ETF'
sector_column = 'Sector'
if ticker_column not in df.columns:
    raise KeyError(f"Column '{ticker_column}' not found in the CSV file. Available columns: {df.columns}")
tickers = df[ticker_column].tolist()
sectors = df[ticker_column].tolist()
etfs = df[etf_column].tolist()
unique_sectors = df['Sector'].unique()
print("Unique Sectors in the Market Cap Stocks CSV:")
print(unique_sectors)


##Filter 2
#Define the list of sector ETFs
sector_etfs = {
    'XLU': 'Utilities',
    'XLB': 'Basic Materials',
    'XLE': 'Energy',
    'XLF': 'Financial Sevicea',
    'XLI': 'Industrials',
    'XLK': 'Technology',
    'XLP': 'Consumer Defensive',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Cyclical',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

#Backtest period
full_start_date = '2019-10-01'
full_end_date = '2023-12-31'

#Fetch Historical Stock Data With 20 & 50 Period EMA
etf_data = {}
for etf in sector_etfs:
    data = yf.download(etf, start=full_start_date, end=full_end_date).copy()
    data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20).copy()
    data['EMA_50'] = talib.EMA(data['Close'], timeperiod=50).copy()
    data['Pct_Change'] = data['Close'].pct_change().copy()
    data['Cumulative_Return'] = ((1 + data['Pct_Change']).cumprod() - 1).copy()
    etf_data[etf] = data

#Filter Trending ETF's
etf_trending = {}
for etf, data in etf_data.items():
    data['EMA_20_Above_EMA_50'] = (data['EMA_20'] > data['EMA_50']).copy() #20 EMA > 50 EMA
    data['Above_EMA_20'] = (data['Close'] > data['EMA_20']).copy() #Price Close Above 20 EMA
    data['Trending_Up'] = ((data['Above_EMA_20'].rolling(window=3).sum() == 3) & data['EMA_20_Above_EMA_50']).copy() #
    etf_trending[etf] = data['Trending_Up']  # Evaluate Historical Data

#DataFrame
trending_df = pd.DataFrame(etf_trending).astype(int) #Boolean values
trending_df.index = pd.to_datetime(trending_df.index)
trending_file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/ETF_Trending.csv' #Save Historical Data
trending_df.to_csv(trending_file_path, index=True)
print(f"ETF Trending Information Saved at: {trending_file_path}")

#Plot Normalized Performance of each Sector ETF
plt.figure(figsize=(14, 8))
for etf, data in etf_data.items():
    plt.plot(data.index, data['Cumulative_Return'].copy(), label=etf)
plt.title('Sector ETF Performance (2020-2024)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend(loc='upper left')
plt.grid(True)
visualization_file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/ETF_Trend_Visualization.png' #Save Visualization
plt.savefig(visualization_file_path)
print(f"ETF Trend Visualization Saved at: {visualization_file_path}")



#Directories
base_data_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting' #Backtest Data
individual_data_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Individual Stock Data' #Individual Stock Data
results_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Results' #Results Data
#Retrieve Yahoo Finance data from each ticker
def fetch_stock_data(ticker, end_date, lookback_start_date): 
    try: #Error detection
        print(f"Fetching data for {ticker}")
        data = yf.download(ticker, start=lookback_start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if data.empty: #Error Detection
            logging.error(f"No data for {ticker}")
            print(f"No data for {ticker}")
            return None
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        print(f"Error fetching data for {ticker}: {e}")
        return None

#Calculate Technical Indicators
def calculate_indicators(data):
    try: #Error Detection
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14) ##Filter 3: ADX (Strength of Trend)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14) ##Filter 4: RSI (Direction)
        data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20) ##Filter 5: EMA (Direction)
        return data
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        print(f"Error calculating indicators: {e}")
        return None

#Create DataFrame 
def create_dataframe(ticker, etf, sector, start_date, end_date, lookback_start_date):
    print(f"Processing {ticker} with ETF {etf} in sector {sector}")  # Error Detection
    trending_file_path = os.path.join(base_data_dir, 'ETF_Trending.csv')  # Access Trending ETF Data
    trending_df = pd.read_csv(trending_file_path, index_col=0, parse_dates=True)  # Get Trending ETF Data
    if etf not in trending_df.columns:  # Error Detection
        print(f"No matching ETF column found for {etf}")
        logging.info(f"No matching ETF column found for {etf}")
        return None
    trending_dates = trending_df[etf]  # Isolate dates where specific ETF's are trending
    if not trending_dates.any():  # Error Detection
        print(f"No trending dates for ETF {etf}")
        logging.info(f"{ticker} not in trending ETF {etf}")
        return None
    data = fetch_stock_data(ticker, end_date, lookback_start_date)  # Include lookback_start_date
    if data is None:  # Error Detection
        return None
    data = calculate_indicators(data)  # Get TA-Lib descriptive data
    if data is None:  # Error Detection
        return None
    data = data.loc[start_date:end_date]  # Only include dates within range
    # Create DataFrame & Add Columns
    df_combined = data.copy()
    df_combined['Ticker'] = ticker
    df_combined['Corresponding ETF'] = etf
    df_combined['Sector'] = sector
    df_combined['SP'] = trending_dates.reindex(data.index, fill_value=0)
    df_combined['Market Cap'] = df[df[ticker_column] == ticker]['Market Cap'].values[0]
    # Add Technical Indicators & Get Previous Trading Day's Data
    df_combined[['ADX', 'RSI', 'EMA_20', 'SP']] = df_combined[['ADX', 'RSI', 'EMA_20', 'SP']].shift(1)
    return df_combined




def backtest_strategy(data):
    initial_balance = 100000  # Initial Capital
    balance = initial_balance
    position = 0  
    buy_price = 0
    trades = []  
    equity_curve = [initial_balance]
    # Iterate over each row in DataFrame
    for index, row in data.iterrows():
        # Buy Stock
        if row['RSI'] > 50 and row['ADX'] > 30 and row['Close'] > row['EMA_20'] and row['Volume'] > 500000 and row['SP'] == 1:  # Conditions to Enter Trade
            if position == 0:  # Ensure not actively in position
                position = balance / row['Open']  # Calculate number of shares to buy
                buy_price = row['Open']  # Record buy price
                balance = 0
                trades.append((index.date(), row['Ticker'], 'Buy', row['Open'], row['Close'], row['RSI'], row['ADX'], row['Volume'], row['EMA_20'], row['Market Cap'], row['Sector'], row['Corresponding ETF'], row['SP'], None, buy_price, index))  # Records trade details
                print(f"Buy on {index.date()}, Price: {buy_price:.2f}")
        # Sell Stock
        elif position > 0:
            if row['RSI'] < 50 or row['ADX'] < 30 or row['Close'] < row['EMA_20']:
                balance = position * row['Close']  # Sell all shares when position is closed
                profit = balance - initial_balance  # Calculate profit
                pct_change = (row['Close'] - buy_price) / buy_price * 100
                duration = (index - trades[-1][-1]).days  # Calculate duration
                trades.append((index.date(), row['Ticker'], 'Sell', row['Open'], row['Close'], row['RSI'], row['ADX'], row['Volume'], row['EMA_20'], row['Market Cap'], row['Sector'], row['Corresponding ETF'], row['SP'], pct_change, row['Close'], duration))  # Records trade details
                position = 0  # No shares held after selling out of position
                print(f"Sell on {index.date()}, Price: {row['Close']:.2f}, Profit: {profit:.2f}, Percentage Change: {pct_change:.2f}%")
        # Update equity curve
        if position > 0:
            equity_curve.append(balance + position * row['Close'])
        else:
            equity_curve.append(balance)

    # Final Balance Calculation
    if position > 0:  # Active trade
        balance = position * data.iloc[-1]['Close']  # Close position at the end
        pct_change = (data.iloc[-1]['Close'] - buy_price) / buy_price * 100  # Calculate percentage change
        duration = (data.index[-1] - trades[-1][-1]).days  # Duration 
        trades.append((data.index[-1].date(), data.iloc[-1]['Ticker'], 'Sell', data.iloc[-1]['Open'], data.iloc[-1]['Close'], data.iloc[-1]['RSI'], data.iloc[-1]['ADX'], data.iloc[-1]['Volume'], data.iloc[-1]['EMA_20'], data.iloc[-1]['Market Cap'], data.iloc[-1]['Sector'], data.iloc[-1]['Corresponding ETF'], data.iloc[-1]['SP'], pct_change, data.iloc[-1]['Close'], duration))  # Records trade details

    total_profit = balance - initial_balance  # Total Profit
    trades_df = pd.DataFrame(trades, columns=['Date', 'Ticker', 'Action', 'Open', 'Close', 'RSI', 'ADX', 'Volume', 'EMA_20', 'Market Cap', 'Sector', 'Corresponding ETF', 'SP', 'Percentage Change', 'Transaction Price', 'Duration'])  # DataFrame to store trades
    trades_df['Date'] = pd.to_datetime(trades_df['Date'])  # Datetime format
    
    # Ensure equity_curve has the same length as data.index
    if len(equity_curve) > len(data):
        equity_curve = equity_curve[:len(data)]
    elif len(equity_curve) < len(data):
        equity_curve.extend([equity_curve[-1]] * (len(data) - len(equity_curve)))

    equity_df = pd.DataFrame({'Date': data.index, 'Equity': equity_curve})
    return total_profit, trades_df, equity_df


# Define process_ticker function
def process_ticker(ticker_info):
    selected_ticker, df, ticker_column, etf_column, sector_column, start_date, end_date, lookback_start_date = ticker_info
    try:
        selected_etf = df[df[ticker_column] == selected_ticker][etf_column].values[0]
        selected_sector = df[df[ticker_column] == selected_ticker][sector_column].values[0]
        stock_df = create_dataframe(selected_ticker, selected_etf, selected_sector, start_date, end_date, lookback_start_date)
        if stock_df is not None:
            stock_df_path = f'/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Individual Stock Data/{selected_ticker}_Data.csv'
            stock_df.to_csv(stock_df_path, index=True)
            print(f"Data for {selected_ticker} saved at {stock_df_path}")
            total_profit, trades_df, _ = backtest_strategy(stock_df)  # Ignore equity_df
            print(f"Total profit from backtesting {selected_ticker}: {total_profit:.2f}")
            initial_balance = 100000
            final_balance = initial_balance + total_profit
            annualized_return = ((final_balance / initial_balance) ** (1 / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25)) - 1) * 100
            print(f"Annualized return: {annualized_return:.2f}%")
            # Debugging: Print columns of trades_df to ensure 'Percentage Change' is included
            print(f"Columns in trades_df for {selected_ticker}: {trades_df.columns}")
            return trades_df, pd.DataFrame()  # Return an empty DataFrame for equity_df
        else:
            print(f"Failed to process {selected_ticker}")
            return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logging.error(f"Error processing ticker {selected_ticker}: {e}")
        print(f"Error processing ticker {selected_ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def main():
    ticker_column = 'Ticker'  # Adjust based on your DataFrame
    etf_column = 'Corresponding ETF'  # Adjust based on your DataFrame
    sector_column = 'Sector'  # Adjust based on your DataFrame

    all_trades_list = []
    results_summary = []
    all_years_trades = pd.DataFrame()  # Initialize an empty DataFrame to collect all trades

    for year in range(2020, 2024):  # Loop through the years 2020, 2021, 2022, 2023
        start_date = pd.to_datetime(f'{year}-01-01')
        end_date = pd.to_datetime(f'{year}-12-31')
        lookback_start_date = start_date - pd.Timedelta(days=90)  # Needed for accurate initial conditions

        all_trades = pd.DataFrame()  # Log Trades

        ticker_info_list = [(ticker, df, ticker_column, etf_column, sector_column, start_date, end_date, lookback_start_date) for ticker in tickers]
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(process_ticker, ticker_info_list)

        for trades_df, _ in results:
            if not trades_df.empty:
                all_trades = pd.concat([all_trades, trades_df], ignore_index=True)

        # Save All Trades DataFrame
        all_trades_file_path = os.path.join(results_dir, f'All_Trades_{year}.csv')
        all_trades.to_csv(all_trades_file_path, index=False)
        print(f"All Trades Data Saved at {all_trades_file_path}")

        # Collect results for summary
        summary = {
            'year': year,
            'all_trades': all_trades,
            'win_trades': None,
            'loss_trades': None,
            'win_loss_ratio': None,
            'average_win_percentage': None,
            'average_loss_percentage': None,
            'average_percentage_change': None,
            'std_dev_percentage_change': None,
            'most_losers_in_row': None,
            'most_winners_in_row': None,
            'total_trades': len(all_trades) // 2,  # Adjust for buy and sell rows
            'total_winners': None,
            'total_losers': None,
            'average_duration': None,  # Added field for average duration
        }

        # Debugging: Print columns of all_trades before accessing 'Percentage Change'
        print(f"Columns in all_trades DataFrame for {year}: {all_trades.columns}")

        # Calculate Descriptive Statistics
        if 'Percentage Change' in all_trades.columns:
            win_trades = all_trades[all_trades['Percentage Change'] > 0]
            loss_trades = all_trades[all_trades['Percentage Change'] <= 0]

            summary['win_trades'] = win_trades
            summary['loss_trades'] = loss_trades
            summary['win_loss_ratio'] = len(win_trades) / len(loss_trades) if len(loss_trades) > 0 else float('inf')
            summary['average_win_percentage'] = win_trades['Percentage Change'].mean()
            summary['average_loss_percentage'] = loss_trades['Percentage Change'].mean()
            summary['average_percentage_change'] = all_trades['Percentage Change'].mean()
            summary['std_dev_percentage_change'] = all_trades['Percentage Change'].std()

            # Calculate most losers in a row
            summary['most_losers_in_row'] = (all_trades['Percentage Change'] <= 0).astype(int).groupby((all_trades['Percentage Change'] > 0).cumsum()).cumsum().max()

            # Calculate most winners in a row
            summary['most_winners_in_row'] = (all_trades['Percentage Change'] > 0).astype(int).groupby((all_trades['Percentage Change'] <= 0).cumsum()).cumsum().max()

            # Total winners and losers
            summary['total_winners'] = len(win_trades)
            summary['total_losers'] = len(loss_trades)

            # Calculate average duration
            summary['average_duration'] = all_trades[all_trades['Duration'].apply(lambda x: isinstance(x, int))]['Duration'].mean()

        results_summary.append(summary)

        # Append results to list for summary
        all_trades_list.append(all_trades)

        # Aggregate all trades across years
        all_years_trades = pd.concat([all_years_trades, all_trades], ignore_index=True)

    # Perform the same calculations on the combined dataset
    summary = {
        'year': 'All Years',
        'all_trades': all_years_trades,
        'win_trades': None,
        'loss_trades': None,
        'win_loss_ratio': None,
        'average_win_percentage': None,
        'average_loss_percentage': None,
        'average_percentage_change': None,
        'std_dev_percentage_change': None,
        'most_losers_in_row': None,
        'most_winners_in_row': None,
        'total_trades': len(all_years_trades) // 2,  # Adjust for buy and sell rows
        'total_winners': None,
        'total_losers': None,
        'average_duration': None,  # Added field for average duration
    }

    # Calculate Descriptive Statistics
    if 'Percentage Change' in all_years_trades.columns:
        win_trades = all_years_trades[all_years_trades['Percentage Change'] > 0]
        loss_trades = all_years_trades[all_years_trades['Percentage Change'] <= 0]

        summary['win_trades'] = win_trades
        summary['loss_trades'] = loss_trades
        summary['win_loss_ratio'] = len(win_trades) / len(loss_trades) if len(loss_trades) > 0 else float('inf')
        summary['average_win_percentage'] = win_trades['Percentage Change'].mean()
        summary['average_loss_percentage'] = loss_trades['Percentage Change'].mean()
        summary['average_percentage_change'] = all_years_trades['Percentage Change'].mean()
        summary['std_dev_percentage_change'] = all_years_trades['Percentage Change'].std()

        # Calculate most losers in a row
        summary['most_losers_in_row'] = (all_years_trades['Percentage Change'] <= 0).astype(int).groupby((all_years_trades['Percentage Change'] > 0).cumsum()).cumsum().max()

        # Calculate most winners in a row
        summary['most_winners_in_row'] = (all_years_trades['Percentage Change'] > 0).astype(int).groupby((all_years_trades['Percentage Change'] <= 0).cumsum()).cumsum().max()

        # Total winners and losers
        summary['total_winners'] = len(win_trades)
        summary['total_losers'] = len(loss_trades)

        # Calculate average duration
        summary['average_duration'] = all_years_trades[all_years_trades['Duration'].apply(lambda x: isinstance(x, int))]['Duration'].mean()

    # Append the summary for all years to results_summary
    results_summary.append(summary)

    # Print Descriptive Statistics for all years
    for summary in results_summary:
        print(f"\nYear: {summary['year']}")
        if summary['win_loss_ratio'] is not None:
            print(f"Win/Loss Ratio: {summary['win_loss_ratio']:.2f}")
            print(f"Average Win Percentage: {summary['average_win_percentage']:.2f}%")
            print(f"Average Loss Percentage: {summary['average_loss_percentage']:.2f}%")
            print(f"Average Percentage Change: {summary['average_percentage_change']:.2f}%")
            print(f"Standard Deviation of Percentage Change: {summary['std_dev_percentage_change']:.2f}%")
            print(f"Most Losers in a Row: {summary['most_losers_in_row']}")
            print(f"Most Winners in a Row: {summary['most_winners_in_row']}")
            print(f"Total Number of Trades: {summary['total_trades']}")
            print(f"Total Winners: {summary['total_winners']}")
            print(f"Total Losers: {summary['total_losers']}")
            print(f"Average Duration: {summary['average_duration']:.2f} days")
        else:
            print(f"'Percentage Change' column not found in all_trades for {summary['year']}")

if __name__ == "__main__":
    main()