#Import Libraries
import pandas as pd  # DataFrame Manipulation
import yfinance as yf  # Descriptive Analysis
import talib  # Technical Analysis
import matplotlib.pyplot as plt  # Visualizations
import logging  # Logging
import os  # Operating System
from multiprocessing import Pool  # Processing
import numpy as np  # Matrices

logging.basicConfig(level=logging.ERROR)

# Define directories
individual_data_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Individual Stock Data'
results_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Results'

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

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

# Process individual stock data files
def process_stock_data_file(file_path, start_date, end_date):
    try:
        stock_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        ticker = os.path.basename(file_path).split('_')[0]
        stock_df['Ticker'] = ticker
        total_profit, trades_df, _ = backtest_strategy(stock_df, start_date, end_date)  # Ignore equity_df
        print(f"Total profit from backtesting {ticker}: {total_profit:.2f}")
        initial_balance = 100000
        final_balance = initial_balance + total_profit
        annualized_return = ((final_balance / initial_balance) ** (1 / ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25)) - 1) * 100
        print(f"Annualized return: {annualized_return:.2f}%")
        return trades_df
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return pd.DataFrame()

def main():
    all_trades_list = []
    results_summary = []
    all_years_trades = pd.DataFrame()  # Initialize an empty DataFrame to collect all trades
    start_date = '2020-01-01'  # Backtest period start date
    end_date = '2023-12-31'  # Backtest period end date

    # Process all stock data files in the directory
    for file_name in os.listdir(individual_data_dir):
        file_path = os.path.join(individual_data_dir, file_name)
        if os.path.isfile(file_path):
            trades_df = process_stock_data_file(file_path, start_date, end_date)
            if not trades_df.empty:
                all_years_trades = pd.concat([all_years_trades, trades_df], ignore_index=True)

    # Save all trades data
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
    spy_data = fetch_stock_data('SPY', start_date, end_date)
    spy_data = spy_data.loc['2020-01-01':]  # Ensure we only use data from 2020 onwards

    # Normalize SPY data to start at the same initial balance
    spy_data['Normalized Close'] = 100000 * (spy_data['Close'] / spy_data['Close'].iloc[0])

    # Create the legend text
    def calculate_annual_growth(portfolio_value, dates):
        annual_growth = {}
        start_year = dates[0].year
        end_year = dates[-1].year
        for year in range(start_year, end_year + 1):
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
