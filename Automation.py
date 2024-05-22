import pandas as pd
import yfinance as yf
import requests
import json


import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Load the CSV file
file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Complete_Stocks.csv'
stock_data = pd.read_csv(file_path)

# Initialize lists to store the fetched data
volumes = []
closes = []
opens = []

# Set the date range for fetching the data
end_date = datetime.today().date()
start_date = end_date - timedelta(days=5)  # Fetch data for the last 5 days to ensure we get yesterday's data

# Iterate over each symbol in the DataFrame and fetch the data
for symbol in stock_data['symbol']:
    try:
        # Fetch historical data
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)
        
        # Check if we have data for yesterday
        if not hist.empty:
            # Get the last available trading day's data
            last_data = hist.iloc[-1]
            volumes.append(last_data['Volume'])
            closes.append(last_data['Close'])
            opens.append(last_data['Open'])
        else:
            # If no data, append NaN
            volumes.append(float('nan'))
            closes.append(float('nan'))
            opens.append(float('nan'))
    except Exception as e:
        # In case of any error, append NaN
        volumes.append(float('nan'))
        closes.append(float('nan'))
        opens.append(float('nan'))
        print(f"Error fetching data for {symbol}: {e}")

# Add the fetched data to the DataFrame
stock_data['Volume'] = volumes
stock_data['Close'] = closes
stock_data['Open'] = opens

# Display the updated DataFrame
print(stock_data.head())

# Save the updated DataFrame to a new CSV file
output_file_path = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Updated_Complete_Stocks.csv'
stock_data.to_csv(output_file_path, index=False)







































# Hypothetical trading platform API endpoints
#TRADING_PLATFORM_API_BASE = 'https://api.tradingplatform.com'
#PLACE_ORDER_ENDPOINT = f'{TRADING_PLATFORM_API_BASE}/order'
##
#Portfolio Parameters
#total_portfolio_value = 1000000  
#max_allocation_percentage = 0.10
#stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ'] #Placeholder Stock Symbols

#Fetch current stock prices from Yahoo Finance
#def fetch_stock_prices(symbols):
   #prices = {}
   # for symbol in symbols:
        #stock = yf.Ticker(symbol)
       # history = stock.history(period='1d')
      #  if not history.empty:
           # prices[symbol] = history['Close'].iloc[0]
       # else:
         #   print(f"No data found for {symbol}, symbol may be incorrect or delisted.")
    #return prices

# Calculate portfolio allocation
#def calculate_allocation(stock_prices):
   # stock_values = {symbol: total_portfolio_value / len(stock_prices) for symbol in stock_prices}
   # portfolio_df = pd.DataFrame(list(stock_values.items()), columns=['Stock', 'Value'])
    #portfolio_df['Price'] = portfolio_df['Stock'].apply(lambda x: stock_prices[x])
    #portfolio_df['Quantity'] = portfolio_df['Value'] / portfolio_df['Price']
    #portfolio_df['Percentage'] = portfolio_df['Value'] / total_portfolio_value

    #exceeds_max = portfolio_df[portfolio_df['Percentage'] > max_allocation_percentage]
   # if not exceeds_max.empty:
       # for index, row in exceeds_max.iterrows():
           # excess_value = row['Value'] - (total_portfolio_value * max_allocation_percentage)
           # portfolio_df.at[index, 'Value'] -= excess_value
           # portfolio_df.at[index, 'Quantity'] = portfolio_df.at[index, 'Value'] / portfolio_df.at[index, 'Price']
          #  portfolio_df.at[index, 'Percentage'] = portfolio_df.at[index, 'Value'] / total_portfolio_value

    #return portfolio_df

# Place orders on the trading platform
#def place_orders(portfolio_df):
   # for index, row in portfolio_df.iterrows():
     #   order_data = {
         #   'symbol': row['Stock'],
        #    'quantity': row['Quantity'],
         #   'price': row['Price'],
         #   'side': 'buy' if row['Value'] > 0 else 'sell'  # Example logic
       # }
    #    response = requests.post(PLACE_ORDER_ENDPOINT, data=json.dumps(order_data), headers={'Content-Type': 'application/json'})
      #  print(f"Order placed for {row['Stock']}: {response.status_code}, {response.text}")

# Main function to run the bot
#def run_bot():
    #stock_prices = fetch_stock_prices(stock_symbols)
   # portfolio_df = calculate_allocation(stock_prices)
   # place_orders(portfolio_df)
    #print(portfolio_df)

# Run the bot
#run_bot()
##