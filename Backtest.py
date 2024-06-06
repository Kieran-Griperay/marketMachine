#Import Libraries
import pandas as pd #DataFrame Manipulation
import yfinance as yf #Descriptive Analysis
import talib #Technical Analysis
import matplotlib.pyplot as plt #Visualizations
from concurrent.futures import ThreadPoolExecutor, as_completed #Parallel Processing
import logging #Logging
import os #Operating System
from multiprocessing import Pool #Processing
import numpy as np #Matrices


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


#Define Sector ETFs
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

#Backtest Periof for ETFs
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
    try:
        print(f"Fetching data for {ticker}")
        data = yf.download(ticker, start=lookback_start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if data.empty:
            raise ValueError(f"No data fetched for {ticker} from {lookback_start_date} to {end_date}")
        print(f"Successfully fetched data for {ticker}")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Calculate Technical Indicators
def calculate_indicators(data):
    try:
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)
        data['EMA_50'] = talib.EMA(data['Close'], timeperiod=50)
        return data
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

# Create DataFrame
def create_dataframe(ticker, etf, sector, start_date, final_end_date, lookback_start_date):
    print(f"Processing {ticker} with ETF {etf} in sector {sector}")
    trending_file_path = os.path.join(base_data_dir, 'ETF_Trending.csv')
    trending_df = pd.read_csv(trending_file_path, index_col=0, parse_dates=True)
    
    if etf not in trending_df.columns:
        print(f"No matching ETF column found for {etf}")
        return None
    
    trending_dates = trending_df[etf]
    if not trending_dates.any():
        print(f"No trending dates for ETF {etf}")
        return None
    
    data = fetch_stock_data(ticker, final_end_date, lookback_start_date)
    if data is None:
        return None
    
    data = calculate_indicators(data)
    if data is None:
        return None
    
    df_combined = data.copy()
    df_combined['Ticker'] = ticker
    df_combined['Corresponding ETF'] = etf
    df_combined['Sector'] = sector
    df_combined['SP'] = trending_dates.reindex(data.index, fill_value=0)
    df_combined['Market Cap'] = df[df[ticker_column] == ticker]['Market Cap'].values[0]
    
    # Shift technical indicators by one day
    df_combined[['ADX', 'RSI', 'EMA_20', 'EMA_50', 'SP']] = df_combined[['ADX', 'RSI', 'EMA_20', 'EMA_50', 'SP']].shift(1)
    
    return df_combined

# Backtest Strategy
def backtest_strategy(data, start_date, end_date):
    initial_balance = 100000  # Initial Capital
    balance = initial_balance
    position = 0  
    buy_price = 0
    trades = []  
    equity_curve = [initial_balance]
    min_start_date = pd.to_datetime(start_date)
    max_end_date = pd.to_datetime(end_date)
    # Iterate over each row in DataFrame
    for index, row in data.iterrows():
        if index < min_start_date:
            continue

        # Buy Stock
        if row['RSI'] > 50 and row['ADX'] > 30 and row['Close'] > row['EMA_20'] and row['SP'] == 1:
            if position == 0:  # Ensure not actively in position
                position = balance / row['Open']  # Calculate number of shares to buy
                buy_price = row['Open']  # Record buy price
                balance = 0
                trades.append((index.date(), row['Ticker'], 'Buy', row['Open'], row['Close'], row['RSI'], row['ADX'], row['Volume'], row['EMA_20'], row['EMA_50'], row['Market Cap'], row['Sector'], row['Corresponding ETF'], row['SP'], None, buy_price, index))  # Records trade details
                print(f"Buy on {index.date()}, Price: {buy_price:.2f}")
        # Sell Stock
        elif position > 0:
            if row['RSI'] < 50 or row['ADX'] < 30 or row['Close'] < row['EMA_20']:
                balance = position * row['Close']  # Sell all shares when position is closed
                profit = balance - initial_balance  # Calculate profit
                pct_change = (row['Close'] - buy_price) / buy_price * 100
                duration = (index - trades[-1][-1]).days  # Calculate duration
                trades.append((index.date(), row['Ticker'], 'Sell', row['Open'], row['Close'], row['RSI'], row['ADX'], row['Volume'], row['EMA_20'], row['EMA_50'], row['Market Cap'], row['Sector'], row['Corresponding ETF'], row['SP'], pct_change, row['Close'], duration))  # Records trade details
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
        trades.append((data.index[-1].date(), data.iloc[-1]['Ticker'], 'Sell', data.iloc[-1]['Open'], data.iloc[-1]['Close'], data.iloc[-1]['RSI'], data.iloc[-1]['ADX'], data.iloc[-1]['Volume'], data.iloc[-1]['EMA_20'], data.iloc[-1]['EMA_50'], data.iloc[-1]['Market Cap'], data.iloc[-1]['Sector'], data.iloc[-1]['Corresponding ETF'], data.iloc[-1]['SP'], pct_change, data.iloc[-1]['Close'], duration))  # Records trade details

    total_profit = balance - initial_balance  # Total Profit
    trades_df = pd.DataFrame(trades, columns=['Date', 'Ticker', 'Action', 'Open', 'Close', 'RSI', 'ADX', 'Volume', 'EMA_20', 'EMA_50', 'Market Cap', 'Sector', 'Corresponding ETF', 'SP', 'Percentage Change', 'Transaction Price', 'Duration'])  # DataFrame to store trades
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
            total_profit, trades_df, _ = backtest_strategy(stock_df, start_date, end_date)  # Ignore equity_df
            print(f"Total profit from backtesting {selected_ticker}: {total_profit:.2f}")
            initial_balance = 100000
            final_balance = initial_balance + total_profit
            annualized_return = ((final_balance / initial_balance) ** (1 / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25)) - 1) * 100
            print(f"Annualized return: {annualized_return:.2f}%")
            return trades_df, pd.DataFrame()  # Return an empty DataFrame for equity_df
        else:
            print(f"Failed to process {selected_ticker}")
            return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logging.error(f"Error processing ticker {selected_ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    ticker_column = 'Ticker'  # Adjust based on your DataFrame
    etf_column = 'Corresponding ETF'  # Adjust based on your DataFrame
    sector_column = 'Sector'  # Adjust based on your DataFrame
    all_trades_list = []
    results_summary = []
    all_years_trades = pd.DataFrame()  # Initialize an empty DataFrame to collect all trades
    lookback_start_date = pd.to_datetime('2019-01-01')  # Adjust based on desired lookback period
    final_end_date = pd.to_datetime('2023-12-31')

    for ticker in tickers:  # Loop through each ticker
        try:
            selected_etf = df[df[ticker_column] == ticker][etf_column].values[0]
            selected_sector = df[df[ticker_column] == ticker][sector_column].values[0]
            stock_df = create_dataframe(ticker, selected_etf, selected_sector, lookback_start_date, final_end_date, lookback_start_date)
            if stock_df is not None:
                stock_df_path = f'/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Individual Stock Data/{ticker}_Data.csv'
                stock_df.to_csv(stock_df_path, index=True)
                print(f"Data for {ticker} saved at {stock_df_path}")

                for year in range(2020, 2024):  # Loop through the years 2020, 2021, 2022, 2023
                    start_date = pd.to_datetime(f'{year}-01-01')
                    end_date = pd.to_datetime(f'{year}-12-31')
                    annual_data = stock_df.loc[start_date:end_date]
                    total_profit, trades_df = backtest_strategy(data, start_date, end_date)  # Ignore equity_df
                    # Collect results for summary
                    summary = {
                        'ticker': ticker,
                        'year': year,
                        'total_profit': total_profit,
                        'trades': trades_df
                    }
                    results_summary.append(summary)
            else:
                print(f"Failed to process {ticker}")
        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}")
            continue

    for year in range(2020, 2024):  # Loop through the years 2020, 2021, 2022, 2023
        start_date = pd.to_datetime(f'{year}-01-01')
        end_date = pd.to_datetime(f'{year}-12-31')
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
            'win_rate': None,
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
            'std_dev_duration': None,
        }

        # Debugging: Print columns of all_trades before accessing 'Percentage Change'
        print(f"Columns in all_trades DataFrame for {year}: {all_trades.columns}")

        # Calculate Descriptive Statistics
        if 'Percentage Change' in all_trades.columns:
            win_trades = all_trades[all_trades['Percentage Change'] > 0]
            loss_trades = all_trades[all_trades['Percentage Change'] <= 0]
            summary['win_trades'] = win_trades
            summary['loss_trades'] = loss_trades
            summary['win_rate'] = len(win_trades) / summary['total_trades'] if summary['total_trades'] > 0 else 0
            summary['average_win_percentage'] = win_trades['Percentage Change'].mean()
            summary['average_loss_percentage'] = loss_trades['Percentage Change'].mean()
            summary['average_percentage_change'] = all_trades['Percentage Change'].mean()
            summary['std_dev_percentage_change'] = all_trades['Percentage Change'].std()
            summary['most_losers_in_row'] = (all_trades['Percentage Change'] <= 0).astype(int).groupby((all_trades['Percentage Change'] > 0).cumsum()).cumsum().max()
            summary['most_winners_in_row'] = (all_trades['Percentage Change'] > 0).astype(int).groupby((all_trades['Percentage Change'] <= 0).cumsum()).cumsum().max()
            summary['total_winners'] = len(win_trades)
            summary['total_losers'] = len(loss_trades)
            summary['average_duration'] = all_trades[all_trades['Duration'].apply(lambda x: isinstance(x, int))]['Duration'].mean()
            summary['std_dev_duration'] = all_trades[all_trades['Duration'].apply(lambda x: isinstance(x, int))]['Duration'].std()
        results_summary.append(summary)
        all_trades_list.append(all_trades)
        all_years_trades = pd.concat([all_years_trades, all_trades], ignore_index=True)
    #Save all trades
    all_years_trades_file_path = os.path.join(results_dir, 'All_Trades_All_Years.csv')
    all_years_trades.to_csv(all_years_trades_file_path, index=False)
    print(f"All Trades Data for All Years Saved at {all_years_trades_file_path}")

    # Perform the same calculations on the combined dataset
    summary = {
        'year': 'All Years',
        'all_trades': all_years_trades,
        'win_trades': None,
        'loss_trades': None,
        'win_rate': None,
        'average_win_percentage': None,
        'average_loss_percentage': None,
        'average_percentage_change': None,
        'std_dev_percentage_change': None,
        'most_losers_in_row': None,
        'most_winners_in_row': None,
        'total_trades': len(all_years_trades) // 2,  # Adjust for buy and sell rows
        'total_winners': None,
        'total_losers': None,
        'average_duration': None,
        'std_dev_duration': None,
    }

    
    if 'Percentage Change' in all_years_trades.columns:
        win_trades = all_years_trades[all_years_trades['Percentage Change'] > 0]
        loss_trades = all_years_trades[all_years_trades['Percentage Change'] <= 0]
        summary['win_trades'] = win_trades
        summary['loss_trades'] = loss_trades
        summary['win_rate'] = len(win_trades) / summary['total_trades'] if summary['total_trades'] > 0 else 0
        summary['average_win_percentage'] = win_trades['Percentage Change'].mean()
        summary['average_loss_percentage'] = loss_trades['Percentage Change'].mean()
        summary['average_percentage_change'] = all_years_trades['Percentage Change'].mean()
        summary['std_dev_percentage_change'] = all_years_trades['Percentage Change'].std()
        summary['most_losers_in_row'] = (all_years_trades['Percentage Change'] <= 0).astype(int).groupby((all_years_trades['Percentage Change'] > 0).cumsum()).cumsum().max()
        summary['most_winners_in_row'] = (all_years_trades['Percentage Change'] > 0).astype(int).groupby((all_years_trades['Percentage Change'] <= 0).cumsum()).cumsum().max()
        summary['total_winners'] = len(win_trades)
        summary['total_losers'] = len(loss_trades)
        summary['average_duration'] = all_years_trades[all_years_trades['Duration'].apply(lambda x: isinstance(x, int))]['Duration'].mean()
        summary['std_dev_duration'] = all_years_trades[all_years_trades['Duration'].apply(lambda x: isinstance(x, int))]['Duration'].std()

    results_summary.append(summary)

    # Print results summary
    for result in results_summary:
        print(f"Year: {result['year']}")
        print(f"Win Rate: {result['win_rate']:.2f}")
        print(f"Average Win Percentage: {result['average_win_percentage']:.2f}%")
        print(f"Average Loss Percentage: {result['average_loss_percentage']:.2f}%")
        print(f"Average Percentage Change: {result['average_percentage_change']:.2f}%")
        print(f"Standard Deviation of Percentage Change: {result['std_dev_percentage_change']:.2f}%")
        print(f"Most Losers in a Row: {result['most_losers_in_row']}")
        print(f"Most Winners in a Row: {result['most_winners_in_row']}")
        print(f"Total Number of Trades: {result['total_trades']}")
        print(f"Total Winners: {result['total_winners']}")
        print(f"Total Losers: {result['total_losers']}")
        print(f"Average Duration: {result['average_duration']:.2f} days")
        print(f"Standard Deviation of Duration: {result['std_dev_duration']:.2f} days")
        print()
    else:
        print(f"'Percentage Change' column not found in all_trades for {summary['year']}")

    # Calculate the average percentage change for each day
    # Calculate the average percentage change for each day
    # Calculate the average percentage change for each day
    all_years_trades['Date'] = pd.to_datetime(all_years_trades['Date'])
    daily_avg_pct_change = all_years_trades.groupby('Date')['Percentage Change'].mean()

# Debugging: Print a few samples of the daily average percentage changes
    print("Sample of daily average percentage changes:")
    print(daily_avg_pct_change.head())

# Check for any extreme values in daily_avg_pct_change
    if daily_avg_pct_change.max() > 100 or daily_avg_pct_change.min() < -100:
        print("Extreme Values Found in Following Daily Average Percentage Changes:")
        print(daily_avg_pct_change[daily_avg_pct_change > 100])
        print(daily_avg_pct_change[daily_avg_pct_change < -100])

# Clamp percentage changes to a reasonable range to avoid extreme impacts
    daily_avg_pct_change = daily_avg_pct_change.clip(lower=-10, upper=10)

# Drop any NaN values in the daily_avg_pct_change
    daily_avg_pct_change = daily_avg_pct_change.dropna()

# Initialize portfolio value
    initial_balance = 100000
    portfolio_value = [initial_balance]
    current_value = initial_balance

# Apply average daily percentage change to calculate portfolio value
    for pct_change in daily_avg_pct_change:
        current_value *= (1 + pct_change / 100)
        portfolio_value.append(current_value)

# Align dates with portfolio values
    dates = [pd.to_datetime('2020-01-01')] + list(daily_avg_pct_change.index)
    if len(dates) > len(portfolio_value):
        dates = dates[:len(portfolio_value)]
    elif len(dates) < len(portfolio_value):
        portfolio_value = portfolio_value[:len(dates)]

# Normalize portfolio values to start at 100000
    portfolio_value = (np.array(portfolio_value) / portfolio_value[0]) * 100000

# Fetch SPY data for comparison
    # Calculate the performance of SPY
    spy_data = fetch_stock_data('SPY', final_end_date, lookback_start_date)
    spy_data = spy_data.loc['2020-01-01':]  # Ensure we only use data from 2020 onwards

# Normalize SPY data to start at the same initial balance
    spy_data['Normalized Close'] = 100000 * (spy_data['Close'] / spy_data['Close'].iloc[0])

# Calculate the average percentage change for each day
    all_years_trades['Date'] = pd.to_datetime(all_years_trades['Date'])
    daily_avg_pct_change = all_years_trades.groupby('Date')['Percentage Change'].mean()

# Debugging: Print a few samples of the daily average percentage changes
    print("Sample of daily average percentage changes:")
    print(daily_avg_pct_change.head())

# Check for any extreme values in daily_avg_pct_change
    if daily_avg_pct_change.max() > 100 or daily_avg_pct_change.min() < -100:
        print("Warning: Extreme values found in daily average percentage changes.")
        print(daily_avg_pct_change[daily_avg_pct_change > 100])
        print(daily_avg_pct_change[daily_avg_pct_change < -100])

# Clamp percentage changes to a reasonable range to avoid extreme impacts
    daily_avg_pct_change = daily_avg_pct_change.clip(lower=-10, upper=10)
    daily_avg_pct_change = daily_avg_pct_change.dropna()

# Initialize portfolio value
    initial_balance = 100000
    portfolio_value = [initial_balance]
    current_value = initial_balance

# Apply average daily percentage change to calculate portfolio value
    for pct_change in daily_avg_pct_change:
        current_value *= (1 + pct_change / 100)
        portfolio_value.append(current_value)

# Align dates with portfolio values
    dates = [pd.to_datetime('2020-01-01')] + list(daily_avg_pct_change.index)
    if len(dates) > len(portfolio_value):
        dates = dates[:len(portfolio_value)]
    elif len(dates) < len(portfolio_value):
        portfolio_value = portfolio_value[:len(dates)]

# Calculate the annual growth rate
    def calculate_annual_growth(portfolio_value, dates):
        annual_growth = {}
        start_year = dates[0].year
        end_year = dates[-1].year
        for year in range(start_year, end_year + 1):
        # Get the first and last value for each year
            start_value = next((v for d, v in zip(dates, portfolio_value) if d.year == year and d == pd.Timestamp(f'{year}-01-01')), None)
            end_value = next((v for d, v in zip(dates, portfolio_value) if d.year == year and d == pd.Timestamp(f'{year}-12-31')), None)
            if start_value is None:
                start_value = next((v for d, v in zip(dates, portfolio_value) if d.year == year), None)
            if end_value is None:
                end_value = next((v for d, v in zip(dates[::-1], portfolio_value[::-1]) if d.year == year), None)
            if start_value is not None and end_value is not None:
                annual_growth[year] = ((end_value - start_value) / start_value) * 100
        return annual_growth

# Get annual growth rates
    annual_growth_rates = calculate_annual_growth(portfolio_value, dates)

# Create the legend text
    legend_text_strategy = 'Strategy:\n' + '\n'.join([f"{year}: {growth_rate:.2f}%" for year, growth_rate in annual_growth_rates.items()])
    legend_text_spy = 'SPY:\n' + '\n'.join([f"{year}: {((spy_data['Normalized Close'].resample('A').last() / spy_data['Normalized Close'].resample('A').first() - 1) * 100).iloc[i]:.2f}%" for i, year in enumerate(range(2020, 2024))])

# Plot the portfolio value and SPY
    plt.figure(figsize=(14, 7))
    plt.plot(dates, portfolio_value, label='Portfolio Value')
    plt.plot(spy_data.index, spy_data['Normalized Close'], label='SPY')
    plt.title('Portfolio Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(100000, max(max(portfolio_value), max(spy_data['Normalized Close'])))  # Set y-axis limits to start at 100,000

# Add the annual growth rates to the legend
    plt.annotate(legend_text_strategy, xy=(0.95, 0.95), xycoords='axes fraction', verticalalignment='top', horizontalalignment='right', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.annotate(legend_text_spy, xy=(0.95, 0.75), xycoords='axes fraction', verticalalignment='top', horizontalalignment='right', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.savefig('/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Results/Portfolio_Performance.png')
    plt.show()

if __name__ == "__main__":
    main()







#2. add more indicators
#3. clean up code
#5. make plot y axis start at 100000
