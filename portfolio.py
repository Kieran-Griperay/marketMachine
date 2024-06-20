import pandas as pd
import numpy as np
import logging
import yfinance as yf
from datetime import datetime
import os
import json

logging.basicConfig(level=logging.INFO)

class Portfolio:
    def __init__(self, initial_balance=100000, portfolio_file='portfolio.json'):
        self.balance = initial_balance
        self.stocks = {}  # Dictionary to hold stock symbol and quantity
        self.transactions = []  # List to hold transactions
        self.history = pd.DataFrame(columns=['Date', 'Balance'])
        self.portfolio_file = portfolio_file
        self.load_portfolio()

    def save_portfolio(self):
        data = {
            'balance': self.balance,
            'stocks': self.stocks,
            'transactions': [(str(t[0]), t[1], t[2], t[3], t[4]) for t in self.transactions],  # Convert datetime to string
            'history': self.history.to_dict()
        }
        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f)
        logging.info(f"Portfolio saved to {self.portfolio_file}")

    def load_portfolio(self):
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)
                self.balance = data['balance']
                self.stocks = data['stocks']
                self.transactions = [(datetime.fromisoformat(t[0]), t[1], t[2], t[3], t[4]) for t in data['transactions']]  # Convert string to datetime
                self.history = pd.DataFrame.from_dict(data['history'])
                logging.info(f"Portfolio loaded from {self.portfolio_file}")

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
        self.update_history()
        self.save_portfolio()
        return True

    def sell_stock(self, ticker, quantity, price):
        if ticker not in self.stocks or self.stocks[ticker] < quantity:
            logging.error(f"Not enough shares to sell {quantity} shares of {ticker}")
            return False
        self.stocks[ticker] -= quantity
        self.balance += quantity * price
        self.transactions.append((datetime.now(), 'SELL', ticker, quantity, price))
        logging.info(f"Sold {quantity} shares of {ticker} at {price} each")
        if self.stocks[ticker] == 0:
            del self.stocks[ticker]
        self.update_history()
        self.save_portfolio()
        return True

    def sell_all_stocks(self, current_prices):
        for ticker in list(self.stocks.keys()):
            quantity = self.stocks[ticker]
            price = current_prices.get(ticker, 0)
            if price > 0:
                self.sell_stock(ticker, quantity, price)

    def get_current_prices(self, tickers):
        current_prices = {}
        try:
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                price = stock.history(period='1d')['Close'].iloc[-1]
                if not np.isnan(price):
                    current_prices[ticker] = price
                    logging.debug(f"Fetched price for {ticker}: {price}")
                else:
                    logging.error(f"No price found for {ticker}.")
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

    def update_history(self):
        current_prices = self.get_current_prices(list(self.stocks.keys()))
        total_value = self.get_portfolio_value(current_prices)
        new_record = pd.DataFrame([{'Date': datetime.now().isoformat(), 'Balance': total_value}])
        self.history = pd.concat([self.history, new_record], ignore_index=True)
        logging.info(f"Portfolio value updated to {total_value}")
        self.save_portfolio()

    def get_portfolio_trend(self):
        return self.history

    def print_portfolio(self):
        print(f"Balance: {self.balance}")
        print("Stocks owned:")
        for ticker, quantity in self.stocks.items():
            print(f"{ticker}: {quantity} shares")

def main():
    portfolio = Portfolio()

    while True:
        print("\nMenu:")
        print("1. View Portfolio")
        print("2. Buy Stocks")
        print("3. Sell Stocks")
        print("4. View Portfolio Value")
        print("5. View Portfolio Trend")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            portfolio.print_portfolio()

        elif choice == '2':
            tickers = pd.read_csv('top_15_stocks.csv')['Ticker'].tolist()
            print(tickers)
            tickers.remove('UNIT')
            tickers.remove('REPL')
            dollar_amount = float(input("Enter dollar amount to invest per stock: "))
            current_prices = portfolio.get_current_prices(tickers)
            for ticker in tickers:
                shares_to_buy = portfolio.calculate_shares_for_amount(ticker, dollar_amount)
                price = current_prices.get(ticker, 0)
                if shares_to_buy > 0 and price > 0:
                    portfolio.buy_stock(ticker, shares_to_buy, price)

        elif choice == '3':
            sell_choice = input("Enter 'all' to sell all shares of every stock or 'single' to sell a quantity of an individual stock: ").lower()
            if sell_choice == 'all':
                tickers = list(portfolio.stocks.keys())
                current_prices = portfolio.get_current_prices(tickers)
                portfolio.sell_all_stocks(current_prices)
            elif sell_choice == 'single':
                ticker = input("Enter ticker symbol: ")
                quantity = int(input("Enter number of shares to sell: "))
                current_prices = portfolio.get_current_prices([ticker])
                price = current_prices.get(ticker, 0)
                if price > 0:
                    portfolio.sell_stock(ticker, quantity, price)
                else:
                    print("Could not execute sell order.")
            else:
                print("Invalid choice. Please try again.")

        elif choice == '4':
            current_prices = portfolio.get_current_prices(list(portfolio.stocks.keys()))
            logging.debug(f"Current Prices: {current_prices}")  # Log the fetched prices
            portfolio_value = portfolio.get_portfolio_value(current_prices)
            print(f"Total Portfolio Value: {portfolio_value}")

        elif choice == '5':
            trend = portfolio.get_portfolio_trend()
            if trend.empty:
                print("No transaction history available.")
            else:
                print(trend)

        elif choice == '6':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
