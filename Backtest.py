#Import Libraries
import pandas as pd  # DataFrame Manipulation
import yfinance as yf  # Descriptive Analysis
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



def backtest_strategy(data, start_date, end_date):
    initial_balance = 100000  # Initial Capital
    balance = initial_balance
    position = 0  
    buy_price = 0
    trades = []  
    equity_curve = [initial_balance]
    min_start_date = pd.to_datetime(start_date)
    max_end_date = pd.to_datetime(end_date)

    # Filter data by date range
    data = data[(data.index >= min_start_date) & (data.index <= max_end_date)].copy()

    # Iterate over each row in DataFrame
    for index, row in data.iterrows():
        # Example Buy/Sell Criteria (customize as needed)
        if row['MACD'] > row['MACD_Signal'] and row['RSI'] < 60 and row['ADX'] > 35:
            if position == 0:  # Ensure not actively in position
                position = balance / row['Open']  # Calculate number of shares to buy
                buy_price = row['Open']  # Record buy price
                balance = 0
                trades.append((index.date(), row['Ticker'], 'Buy', row['Open'], row['Close'], row['RSI'], row['ADX'], row['Volume'], 
                               row['EMA_20'], row['EMA_50'], row['Upper_BBand'], row['Middle_BBand'], row['Lower_BBand'], 
                               row['MACD'], row['MACD_Signal'], row['MACD_Hist'], row['ATR'], row['CCI'], row['Williams_%R'], 
                               row['MFI'], row['OBV'], row['Market Cap'], row['Sector'], row['Corresponding ETF'], row['SP'], 
                               None, buy_price, index))  # Record trade details
                print(f"Buy on {index.date()}, Price: {buy_price:.2f}")

        elif position > 0:
            if row['MACD'] < row['MACD_Signal'] and row['RSI'] > 60:
                balance = position * row['Close']  # Sell all shares when position is closed
                profit = balance - initial_balance  # Calculate profit
                pct_change = (row['Close'] - buy_price) / buy_price * 100
                duration = (index - trades[-1][-1]).days  # Calculate duration
                trades.append((index.date(), row['Ticker'], 'Sell', row['Open'], row['Close'], row['RSI'], row['ADX'], row['Volume'], 
                               row['EMA_20'], row['EMA_50'], row['Upper_BBand'], row['Middle_BBand'], row['Lower_BBand'], 
                               row['MACD'], row['MACD_Signal'], row['MACD_Hist'], row['ATR'], row['CCI'], row['Williams_%R'], 
                               row['MFI'], row['OBV'], row['Market Cap'], row['Sector'], row['Corresponding ETF'], row['SP'], 
                               pct_change, row['Close'], duration))  # Record trade details
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
        trades.append((data.iloc[-1].date(), data.iloc[-1]['Ticker'], 'Sell', data.iloc[-1]['Open'], data.iloc[-1]['Close'], 
                       data.iloc[-1]['RSI'], data.iloc[-1]['ADX'], data.iloc[-1]['Volume'], data.iloc[-1]['EMA_20'], 
                       data.iloc[-1]['EMA_50'], data.iloc[-1]['Upper_BBand'], data.iloc[-1]['Middle_BBand'], 
                       data.iloc[-1]['Lower_BBand'], data.iloc[-1]['MACD'], data.iloc[-1]['MACD_Signal'], 
                       data.iloc[-1]['MACD_Hist'], data.iloc[-1]['ATR'], data.iloc[-1]['CCI'], data.iloc[-1]['Williams_%R'], 
                       data.iloc[-1]['MFI'], data.iloc[-1]['OBV'], data.iloc[-1]['Market Cap'], data.iloc[-1]['Sector'], 
                       data.iloc[-1]['Corresponding ETF'], data.iloc[-1]['SP'], pct_change, data.iloc[-1]['Close'], duration))  # Record trade details

    total_profit = balance - initial_balance  # Total Profit
    trades_df = pd.DataFrame(trades, columns=['Date', 'Ticker', 'Action', 'Open', 'Close', 'RSI', 'ADX', 'Volume', 'EMA_20', 
                                              'EMA_50', 'Upper_BBand', 'Middle_BBand', 'Lower_BBand', 'MACD', 'MACD_Signal', 
                                              'MACD_Hist', 'ATR', 'CCI', 'Williams_%R', 'MFI', 'OBV', 'Market Cap', 'Sector', 
                                              'Corresponding ETF', 'SP', 'Percentage Change', 'Transaction Price', 'Duration'])  # DataFrame to store trades
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



# Define the main function
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


# Example: Assuming `all_years_trades` is a DataFrame with 'Date', 'Action', 'Percentage Change', and 'Transaction Price' columns
# Replace this line with your actual data
# all_years_trades = pd.read_csv('all_years_trades.csv')

# Convert 'Date' to datetime
   

# Assuming `all_years_trades` is already loaded with the necessary data

# Convert 'Date' to datetime

# Convert 'Date' to datetime

    
# Convert 'Date' to datetime
    all_years_trades['Date'] = pd.to_datetime(all_years_trades['Date'])

# Filter for Sell actions
    sell_trades = all_years_trades[all_years_trades['Action'] == 'Sell']

# Sort by date to ensure chronological order
    sell_trades = sell_trades.sort_values(by='Date')

# Initialize portfolio value
    initial_balance = 100000
    portfolio_value = [initial_balance]
# Initialize dataframe to save results
    results = []

# Calculate the weighted daily average percentage change
    def calculate_weighted_avg_pct_change(trades, initial_balance, max_position_percent=5):
        current_value = initial_balance # Make sure current_value is accessible here
        daily_pct_changes = []
        dates = trades['Date'].unique()

        for date in dates:
            daily_trades = trades[trades['Date'] == date]
            num_trades = len(daily_trades)

        # Print details for each day with "Sell" actions
            print(f"Date: {date.strftime('%Y-%m-%d')}")
            print(f"Trades closed: {num_trades}")
        
        # Calculate the impact of each trade on the portfolio
            daily_impact = 0
            for _, trade in daily_trades.iterrows():
                if pd.isna(trade['Percentage Change']):
                    print(f"Skipping trade with NaN Percentage Change: {trade['Ticker']}")
                    continue  # Skip trades with NaN Percentage Change
            
            # Calculate the position size value, adjusted for number of trades
                adjusted_position_size_percent = max_position_percent / max(num_trades, 1)  # Avoid division by zero
                position_size_value = (adjusted_position_size_percent / 100) * current_value
                weighted_change = (trade['Percentage Change'] / 100) * position_size_value
                daily_impact += weighted_change

        # Convert impact to percentage of the portfolio
            daily_pct_change = (daily_impact / current_value) * 100
            daily_pct_changes.append(daily_pct_change)

        # Print the calculated daily percentage change
            print(f"Account Percentage Change: {daily_pct_change:.2f}%\n")

        # Update the current value for the next iteration
            current_value += daily_impact

        # Save results for CSV export
            results.append({
                'Date': date,
                'Trades closed': num_trades,
                'Account Percentage Change': daily_pct_change,
                'Updated Portfolio Value': current_value
            })

        return pd.Series(daily_pct_changes, index=dates), current_value
    
    daily_avg_pct_change, current_value = calculate_weighted_avg_pct_change(sell_trades, initial_balance)

# Drop any NaN values
    daily_avg_pct_change = daily_avg_pct_change.dropna()
    df_results = pd.DataFrame(results)
    df_results.to_csv('/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Results/Daily_Percentage_Changes.csv', index=False)

    mean_change = df_results['Account Percentage Change'].mean()
    std_change = df_results['Account Percentage Change'].std()

# Determine outliers based on 3 standard deviations
    threshold = 3 * std_change
    outliers = df_results[
        (df_results['Account Percentage Change'] > (mean_change + threshold)) |
        (df_results['Account Percentage Change'] < (mean_change - threshold))
    ]

# Print outlier details
    print("Outliers in Daily Percentage Change:")
    print(outliers)
    print(f"Number of outliers: {len(outliers)}")
    print(f"Threshold to be classified as an outlier: ±{threshold:.2f}%")
    

# Calculate the daily average percentage changes for "Sell" actions only
    

# Apply the daily average percentage changes to the portfolio value
    filtered_results = df_results[
        (df_results['Account Percentage Change'] <= (mean_change + threshold)) &
        (df_results['Account Percentage Change'] >= (mean_change - threshold))
    ]

    current_value = initial_balance
    portfolio_value = [current_value]

    for _, row in filtered_results.iterrows():
        pct_change = row['Account Percentage Change']
        daily_dollar_change = (pct_change / 100) * current_value
        current_value += daily_dollar_change
        portfolio_value.append(current_value)


# Align dates with portfolio values
    dates = [pd.to_datetime('2020-01-01')] + list(daily_avg_pct_change.index)
    if len(dates) > len(portfolio_value):
        dates = dates[:len(portfolio_value)]
    elif len(dates) < len(portfolio_value):
        portfolio_value = portfolio_value[:len(dates)]

# Format portfolio values for the plot
    portfolio_value = np.array(portfolio_value)

# Create the legend text for annual growth rates
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
    legend_text_strategy = 'Strategy:\n' + '\n'.join([f"{year}: {growth_rate:.2f}%" for year, growth_rate in annual_growth_rates.items()])



    #SPY Normalized
    spy_ticker = yf.Ticker("SPY")
    spy_data = spy_ticker.history(period="max")
    spy_data.index = spy_data.index.tz_localize(None)
    spy_data = spy_data[(spy_data.index >= dates[0]) & (spy_data.index <= dates[-1])]
    initial_spy_value = spy_data.iloc[0]['Close']
    spy_scaled = (spy_data['Close'] / initial_spy_value) * initial_balance









# Plot the portfolio value
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(dates, portfolio_value, label='Portfolio Value', color='blue')
    ax1.plot(spy_data.index, spy_scaled, label='SPY', color='green')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(min(portfolio_value), max(portfolio_value))  # Setting y-axis range
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))  # Format y-axis to display as whole numbers with commas
    ax1.grid(True)

    fig.suptitle('Portfolio Performance Over Time Compared to SPY')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# Add the annual growth rates to the legend
    plt.annotate(legend_text_strategy, xy=(0.95, 0.95), xycoords='axes fraction', verticalalignment='top', horizontalalignment='right', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.savefig('/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Results/Portfolio_Performance.png')
    plt.show()

# Save the results to a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv('/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Results/Daily_Percent_Changes.csv', index=False)

# Define the main functio



if __name__ == "__main__":
    main()




#current_value = initial_balance  # Re-initialize current_value for applying daily changes
    #for pct_change in daily_avg_pct_change:
        #daily_dollar_change = (pct_change / 100) * current_value
        #current_value += daily_dollar_change
       # portfolio_value.append(current_value)