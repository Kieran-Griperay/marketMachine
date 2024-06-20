#Import Libraries
import pandas as pd #DataFrame Manipulation
import os #Operating Systems
from pandas.tseries.offsets import BDay #Business Day Calculations
from multiprocessing import Pool #Parallel Processing

#Directories
stock_data_folder = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Individual Stock Data'  
results_folder = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Individual Stock Results'  
final_results_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Total Results/All_Results.csv'  
os.makedirs(results_folder, exist_ok=True) 
#Required Columns For Backtest
required_columns = ['Date', 'Open', 'Close', 'EMA_20', 'EMA_50', 'ADX', 'RSI', 'Sector', 'Volume', 'Upper_BBand', 'Middle_BBand', 'Lower_BBand', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'CCI', 'Williams_%R', 'MFI', 'OBV', 'Corresponding ETF', 'SP', 'Market Cap']  # List of required columns for analysis


#Backtest Function
def backtest_stock(file_path):
    df = pd.read_csv(file_path) #Read Stock Data from CSV File
    if not all(col in df.columns for col in required_columns): #Ensure all Required Columns are Present
        return None
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') #Convert 'Date' Column to Datetime
    df = df.sort_values(by='Date') #Sort 
    df = df.dropna(subset=required_columns) #Remove Rows with Missing Values 
    position = 0 #Initialize Position
    transactions = [] #Initialize List
    for i in range(1, len(df) - 1):
        if (df.loc[i, 'EMA_20'] > df.loc[i, 'EMA_50'] and #Buy Condition 1
            df.loc[i, 'ADX'] > 30 and #Buy Condition 2
            df.loc[i, 'RSI'] > 50): #Buy Condition 3
            if position == 0: #Ensure Not Already in a Position
                position = 1  #Enter Buy Position
                buy_price = df.loc[i + 1, 'Open'] #Set Buy Price to Next Day's Open Price
                transaction = {'Date': df.loc[i + 1, 'Date'], 'Price': buy_price, 'Signal': 'Buy'} #Create Buy Transaction Record
                for col in required_columns: #Add Required Columns 
                    transaction[col] = df.loc[i + 1, col]
                transactions.append(transaction) #Append Transaction to List
        elif (df.loc[i, 'EMA_20'] <= df.loc[i, 'EMA_50'] or #Sell Condition 1
              df.loc[i, 'ADX'] <= 30 or #Sell Condition 2
              df.loc[i, 'RSI'] <= 50): #Sell Condition 3
            if position == 1:  #Ensure Already in a Position
                position = 0  #Exit Buy Position
                sell_price = df.loc[i, 'Close'] #Set Sell Price to Today's Close Price
                transaction = {'Date': df.loc[i, 'Date'], 'Price': sell_price, 'Signal': 'Sell'} #Create Sell Transaction Record
                for col in required_columns: #Add Required Columns 
                    transaction[col] = df.loc[i, col]
                transactions.append(transaction) #Append Transaction to List
    if not transactions: 
        return None
    transactions_df = pd.DataFrame(transactions).sort_values(by='Date') #Create DataFrame 
    stock_name = os.path.basename(file_path).replace('_Data.csv', '') #Unique Ticker
    transactions_df['Stock'] = stock_name #Add Ticker 

    #Filter Buys Without Corresponding Sells
    buy_indices = transactions_df[transactions_df['Signal'] == 'Buy'].index #Get Indices of Buy Transactions
    sell_indices = transactions_df[transactions_df['Signal'] == 'Sell'].index #Get Indices of Sell Transactions
    if len(sell_indices) < len(buy_indices): #Ensure Buy Transactions Have Sorresponding Sell Transactions
        buy_indices = buy_indices[:len(sell_indices)] #Trim if Needed
    transactions_df = transactions_df.loc[buy_indices.union(sell_indices)].sort_values(by='Date').reset_index(drop=True) #Filter & Sort Transactions
    transactions_df['Percentage Change'] = None #Initialize
    transactions_df['Duration'] = None #Initialize 
    for buy_idx, sell_idx in zip(buy_indices, sell_indices):
        buy_price = transactions_df.loc[buy_idx, 'Price'] #Get Buy Price
        sell_price = transactions_df.loc[sell_idx, 'Price'] #Get Sell Price
        percentage_change = ((sell_price - buy_price) / buy_price) * 100 #Calculate Percentage Change
        transactions_df.at[sell_idx, 'Percentage Change'] = round(percentage_change, 2) #Assign Percentage Change to Sell Transaction
        buy_date = transactions_df.loc[buy_idx, 'Date'] #Get Buy Date
        sell_date = transactions_df.loc[sell_idx, 'Date'] #Get Sell Date
        duration = len(pd.date_range(buy_date, sell_date, freq=BDay())) #Calculate Duration in Business Days
        transactions_df.at[sell_idx, 'Duration'] = duration #Assign Duration to Sell Transaction
    transactions_df['Price'] = transactions_df['Price'].round(2) #Round 
    transactions_df['Percentage Change'] = transactions_df['Percentage Change'].round(2) #Round
    return transactions_df  

def process_file(file_name):
    stock_path = os.path.join(stock_data_folder, file_name) #Construct Full File Path
    results = backtest_stock(stock_path) #Run Backtest on Stock Data
    if results is not None and not results.empty: #Check for Valid Results
        result_file_path = os.path.join(results_folder, f"{file_name.replace('.csv', '')}_backtest_results.csv") #Construct Result File Path
        results.to_csv(result_file_path, index=False) #Save Results to CSV 
        return results  
    return pd.DataFrame() 

if __name__ == "__main__": 
    stock_files = [file for file in os.listdir(stock_data_folder) if file.endswith('.csv')] #List all CSV Files in Stock Data Folder
    with Pool() as pool: #Use Multiprocessing 
        all_results = pool.map(process_file, stock_files) #Map 
    all_results = [df for df in all_results if df is not None and not df.empty] #Filter out Empty Dataframes
    if all_results: #Check for Results
        combined_results = pd.concat(all_results, ignore_index=True) #Concatenate all Results 
        combined_results.to_csv(final_results_path, index=False) #Save 

        #Descriptive Statistics of Strategy Performance
        avg_pct_change = combined_results['Percentage Change'].mean() 
        avg_duration = combined_results['Duration'].mean()  
        print(f"Average Percentage Change: {avg_pct_change:.2f}%")  
        print(f"Average Duration (in business days): {avg_duration:.2f} days")  
        total_days = (combined_results['Date'].max() - combined_results['Date'].min()).days  
        avg_trades_per_day = len(combined_results) / total_days  
        print(f"Average Number of Trades Suggested Per Day: {avg_trades_per_day:.2f}")
        sector_breakdown = combined_results['Sector'].value_counts(normalize=True) * 100  
        print("Percentage Breakdown of Trades by Sector:")  
        print(sector_breakdown.to_string())  
        print(f"Total Number of Trades: {len(combined_results) // 2}")  
        print(f"Total Stocks Analyzed: {len(stock_files)}")  
        print(f"Total Unique Stocks Traded: {combined_results['Stock'].nunique()}") 


