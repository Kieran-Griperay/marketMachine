import pandas as pd
import numpy as np
import logging
import yfinance as yf
from datetime import datetime

logging.basicConfig(level=logging.INFO)

class Portfolio:
    def __init__(self, initial_balance=100000):
        self.balance = initial_balance
        self.stocks = {}  # Dictionary to hold stock symbol and quantity
        self.transactions = []  # List to hold transactions
        self.history = pd.DataFrame(columns=['Date', 'Balance'])

    def buy_stock(self, ticker, quantity, price):
        if price == 0:
            logging.error(f"Price for {ticker} is not available. Skipping purchase.")
            return False

        total_cost = quantity * price
        if total_cost > self.balance:
            logging.error(f"Not enough balance to buy {quantity} shares of {ticker}")
            return False
        self.balance -= total_cost
        if ticker in self.stocks:
            self.stocks[ticker] += quantity
        else:
            self.stocks[ticker] = quantity
        self.transactions.append((datetime.now(), 'BUY', ticker, quantity, price))
        logging.info(f"Bought {quantity} shares of {ticker} at {price} each")
        return True

    def sell_stock(self, ticker, quantity, price):
        if ticker not in self.stocks or self.stocks[ticker] < quantity:
            logging.error(f"Not enough shares to sell {quantity} shares of {ticker}")
            return False
        self.stocks[ticker] -= quantity
        self.balance += quantity * price
        self.transactions.append((datetime.now(), 'SELL', ticker, quantity, price))
        logging.info(f"Sold {quantity} shares of {ticker} at {price} each")
        return True

    def get_current_prices(self, tickers):
        current_prices = {}
        try:
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                price = stock.history(period='1d')['Close'][-1]
                if not np.isnan(price):
                    current_prices[ticker] = price
            return current_prices
        except Exception as e:
            logging.error(f"Error fetching current prices: {e}")
            return {}

    def calculate_shares_for_amount(self, ticker, dollar_amount):
        price = self.get_current_prices([ticker]).get(ticker, 0)
        if price > 0:
            return dollar_amount // price
        else:
            logging.error(f"Could not fetch price for {ticker}")
            return 0

    def get_portfolio_value(self, current_prices):
        total_value = self.balance
        for ticker, quantity in self.stocks.items():
            total_value += quantity * current_prices.get(ticker, 0)
        return total_value

    def update_history(self, current_prices):
        total_value = self.get_portfolio_value(current_prices)
        new_record = pd.DataFrame([{'Date': datetime.now(), 'Balance': total_value}])
        self.history = pd.concat([self.history, new_record], ignore_index=True)
        logging.info(f"Portfolio value updated to {total_value}")

    def get_portfolio_trend(self):
        return self.history

    def print_portfolio(self):
        print(f"Balance: {self.balance}")
        print("Stocks owned:")
        for ticker, quantity in self.stocks.items():
            print(f"{ticker}: {quantity} shares")

# Example usage
if __name__ == "__main__":
    portfolio = Portfolio()

    # Simulating some stock tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'ALGN', 'CYRX', 'EVH', 'FTRE', 'LAW', 'NAMS', 'RELY', 'REPL', 'U', 'UNIT']

    # Fetching current prices
    current_prices = portfolio.get_current_prices(tickers)
    print("Current Prices:", current_prices)

    # Buying stocks with a dollar amount
    for ticker in tickers:
        dollar_amount = 1000  # Example amount
        shares_to_buy = portfolio.calculate_shares_for_amount(ticker, dollar_amount)
        price = current_prices.get(ticker, 0)
        if shares_to_buy > 0 and price > 0:
            portfolio.buy_stock(ticker, shares_to_buy, price)

    # Selling stocks
    if 'AAPL' in current_prices and portfolio.stocks.get('AAPL', 0) >= 5:
        portfolio.sell_stock('AAPL', 5, current_prices['AAPL'])

    # Updating portfolio history
    portfolio.update_history(current_prices)

    # Printing portfolio
    portfolio.print_portfolio()

    # Getting portfolio trend
    trend = portfolio.get_portfolio_trend()
    print(trend)
