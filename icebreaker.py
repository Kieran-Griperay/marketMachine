import requests
import pandas as pd
import numpy as np

# Fetch order book data from Coinbase Pro API
def fetch_order_book(symbol='BTC-USD', level=2):
    api_url = f"https://api.pro.coinbase.com/products/{symbol}/book"
    params = {
        'level': level
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        order_book = response.json()
        return order_book
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

# Parse order book data into a DataFrame
def parse_order_book(order_book):
    bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity', 'num_orders'])
    asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity', 'num_orders'])
    
    bids['price'] = pd.to_numeric(bids['price'])
    bids['quantity'] = pd.to_numeric(bids['quantity'])
    bids['num_orders'] = pd.to_numeric(bids['num_orders'])
    asks['price'] = pd.to_numeric(asks['price'])
    asks['quantity'] = pd.to_numeric(asks['quantity'])
    asks['num_orders'] = pd.to_numeric(asks['num_orders'])

    bids['side'] = 'bid'
    asks['side'] = 'ask'
    
    order_book_df = pd.concat([bids, asks])
    order_book_df['timestamp'] = pd.Timestamp.now()  # Add a timestamp column
    
    return order_book_df

# Enhanced iceberg detection function with detailed debugging
def detect_iceberg(df, quantity_threshold=0.01, repetition_threshold=2, time_window='10s'):
    """
    Detect potential iceberg trades in order book data.
    
    Args:
    df (pd.DataFrame): DataFrame with order book data
    quantity_threshold (float): Minimum quantity to consider for iceberg detection
    repetition_threshold (int): Minimum number of repetitive orders to consider as iceberg
    time_window (str): Time window to consider for repetition, e.g., '10s', '1m'
    
    Returns:
    pd.DataFrame: DataFrame with detected iceberg orders
    """
    iceberg_orders = []
    current_orders = {}
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for index, row in df.iterrows():
        if row['quantity'] < quantity_threshold:
            continue
        
        key = (row['price'], row['side'])
        if key not in current_orders:
            current_orders[key] = [row['timestamp']]
        else:
            current_orders[key].append(row['timestamp'])
            # Check if orders fall within the specified time window
            recent_orders = [t for t in current_orders[key] if (row['timestamp'] - t).total_seconds() <= pd.Timedelta(time_window).total_seconds()]
            current_orders[key] = recent_orders
            
            # Debugging: Print current state of current_orders
            print(f"Debug: Current Orders at {key} = {len(current_orders[key])} within {time_window}")
            
            if len(current_orders[key]) == repetition_threshold:
                iceberg_orders.append(row)

    return pd.DataFrame(iceberg_orders)

# Backtesting the detection algorithm
def backtest_iceberg_detection(df, iceberg_orders):
    """
    Backtest the iceberg detection algorithm.
    
    Args:
    df (pd.DataFrame): Original order book data
    iceberg_orders (pd.DataFrame): Detected iceberg orders
    
    Returns:
    pd.DataFrame: Results of backtesting
    """
    backtest_results = []

    for _, iceberg_order in iceberg_orders.iterrows():
        timestamp = iceberg_order['timestamp']
        price = iceberg_order['price']
        side = iceberg_order['side']
        
        # Check the impact on price after the iceberg order detection
        future_prices = df[(df['timestamp'] > timestamp) & (df['side'] == side)]['price']
        
        if side == 'bid':
            price_change = future_prices.max() - price  # Looking for price increase
        else:
            price_change = price - future_prices.min()  # Looking for price decrease
        
        backtest_results.append({
            'timestamp': timestamp,
            'price': price,
            'side': side,
            'price_change': price_change
        })

    return pd.DataFrame(backtest_results)

# Main execution
if __name__ == "__main__":
    # Fetch and parse order book data
    order_book = fetch_order_book()
    order_book_df = parse_order_book(order_book)

    # Debug: Print fetched order book data
    print("Order Book Data:")
    print(order_book_df.head(10))

    # Detect iceberg orders
    iceberg_orders = detect_iceberg(order_book_df, quantity_threshold=0.01, repetition_threshold=2, time_window='10s')

    # Debug: Print detected iceberg orders
    print("Detected Iceberg Orders:")
    print(iceberg_orders)

    # Perform backtesting
    backtest_results = backtest_iceberg_detection(order_book_df, iceberg_orders)

    # Debug: Print backtest results
    print("Backtest Results:")
    print(backtest_results)
