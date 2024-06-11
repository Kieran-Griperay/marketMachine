import pandas as pd
import logging
import os

logging.basicConfig(level=logging.ERROR)

# Define directories
individual_data_dir = 'Stock Filtration/Individual Stock Data'
results_dir = 'Stock Filtration/Backtesting/Results'

def filter_best_stocks(data):
    try:
        # Define criteria for "best" stocks (customize as needed)
        best_stocks = data[(data['MACD'] > data['MACD_Signal']) & 
                           (data['RSI'] < 60) & 
                           (data['ADX'] > 35)]
        return best_stocks
    except Exception as e:
        logging.error(f"Error filtering best stocks: {e}")
        return pd.DataFrame()

def process_stock_data_file(file_path):
    try:
        stock_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        ticker = os.path.basename(file_path).split('_')[0]
        stock_df['Ticker'] = ticker
        
        # Get the most recent data
        most_recent_data = stock_df.iloc[-1:]
        
        best_stocks = filter_best_stocks(most_recent_data)
        return best_stocks[['Ticker']] if not best_stocks.empty else pd.DataFrame()
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return pd.DataFrame()

def main():
    all_best_stocks = pd.DataFrame()

    # Process all stock data files in the directory
    for file_name in os.listdir(individual_data_dir):
        file_path = os.path.join(individual_data_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            best_stocks = process_stock_data_file(file_path)
            if not best_stocks.empty:
                all_best_stocks = pd.concat([all_best_stocks, best_stocks], ignore_index=True)

    # Save or display the best stocks for today
    best_stocks_file_path = os.path.join(results_dir, 'Best_Stocks_Today.csv')
    all_best_stocks[['Ticker']].drop_duplicates().to_csv(best_stocks_file_path, index=False)
    print(f"Best Stocks for Today Saved at {best_stocks_file_path}")
    print(all_best_stocks[['Ticker']].drop_duplicates())

if __name__ == "__main__":
    main()
