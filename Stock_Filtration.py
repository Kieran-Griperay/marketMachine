# Stock_Filtration.py

import yfinance as yf
import pandas as pd
import datetime
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import talib

def fetch_data(ticker, yesterday, today_str):
    print(f"Processing Ticker: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=(yesterday - timedelta(days=50)).strftime('%Y-%m-%d'), end=today_str)
        if not hist.empty and len(hist) >= 35:
            print(f"Data Fetched for {ticker}.")
            hist['symbol'] = ticker
            hist['marketCap'] = stock.info.get('marketCap', 'N/A')
            hist['sector'] = stock.info.get('sector', 'N/A')
            hist['averageVolume10days'] = stock.info.get('averageVolume10days', 'N/A')
            hist['ADX'] = talib.ADX(hist['High'], hist['Low'], hist['Close'], timeperiod=14)
            hist['RSI'] = talib.RSI(hist['Close'], timeperiod=14)
            last_day_data = hist.iloc[-1:]
            return last_day_data
        else:
            print(f"No data available for {ticker}.")
            return None
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        return None

def load_and_filter_data(file_path):
    tickers_df = pd.read_csv(file_path)
    tickers = tickers_df['symbol'].tolist()

    yesterdays_data = pd.DataFrame()
    now = datetime.now()
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now > market_close_time:
        yesterday = now
    else:
        yesterday = now - timedelta(days=1)
    if yesterday.weekday() == 5:
        yesterday = yesterday - timedelta(days=1)
    elif yesterday.weekday() == 6:
        yesterday = yesterday - timedelta(days=2)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    today_str = now.strftime('%Y-%m-%d')

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(fetch_data, ticker, yesterday, today_str): ticker for ticker in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                yesterdays_data = pd.concat([yesterdays_data, result])

    yesterdays_data.reset_index(inplace=True)
    yesterdays_data['marketCap'] = pd.to_numeric(yesterdays_data['marketCap'], errors='coerce')
    yesterdays_data['averageVolume10days'] = pd.to_numeric(yesterdays_data['averageVolume10days'], errors='coerce')
    yesterdays_data = yesterdays_data[['Date', 'Open', 'Close', 'Volume', 'symbol', 'marketCap', 'sector', 'averageVolume10days', 'ADX', 'RSI']]
    return yesterdays_data

def filter_stocks(file_path):
    yesterdays_data = load_and_filter_data(file_path)
    filtered_by_market_cap = yesterdays_data[yesterdays_data['marketCap'] > 300000000]
    filtered_by_volume = filtered_by_market_cap[filtered_by_market_cap['averageVolume10days'] > 500000].copy()
    etf_symbols = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLU', 'XLC']
    etf_data = {symbol: yf.Ticker(symbol).history(period="3mo")['Close'] for symbol in etf_symbols}
    etf_df = pd.DataFrame(etf_data)
    returns = (etf_df.iloc[-1] / etf_df.iloc[0] - 1) * 100
    top_3_etfs = returns.sort_values(ascending=False).head(3)
    etf_to_sector = {
        'XLY': 'Consumer Cyclical',
        'XLP': 'Consumer Defensive',
        'XLE': 'Energy',
        'XLF': 'Financial Services',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLK': 'Technology',
        'XLB': 'Basic Materials',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities',
        'XLC': 'Communication Services'
    }
    top_sectors = [etf_to_sector[etf] for etf in top_3_etfs.index if etf in etf_to_sector]
    filtered_by_sector = filtered_by_volume[filtered_by_volume['sector'].isin(top_sectors)].reset_index().sort_values(by='marketCap', ascending=False)
    filtered_by_adx = filtered_by_sector[filtered_by_sector['ADX'] > 35]
    filtered_by_rsi = filtered_by_adx[filtered_by_adx['RSI'] > 50]

    filtered_symbols = filtered_by_rsi['symbol'].tolist()
    pd.DataFrame(filtered_symbols, columns=['symbol']).to_csv('filtered_symbols.csv', index=False)
    return filtered_symbols

if __name__ == "__main__":
    file_path = 'Merged_Stocks.csv'
    filtered_symbols = filter_stocks(file_path)
    print(f"Filtered symbols: {filtered_symbols}")
