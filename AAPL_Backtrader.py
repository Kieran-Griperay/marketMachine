import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from datetime import timedelta

# Define directories
results_dir = '/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Backtesting/Results'

# Define the Strategy
class EMACrossoverStrategy(bt.Strategy):
    params = dict(
        ema_short_period=20,
        ema_long_period=50,
        stake_percentage=0.1,  # Use 10% of the portfolio for each trade
        commission_rate=0.001  # 0.1% commission
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open

        # Initialize EMAs
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema_short_period)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.ema_long_period)

        self.order = None
        self.trades = []  # List to store trade details
        self.initial_portfolio_value = self.broker.get_cash()
        self.running_portfolio_balance = self.initial_portfolio_value
        self.buy_date = None  # Track the date of the last buy order
        self.buy_shares = 0  # Track the number of shares bought
        self.portfolio_value_history = []  # Track the portfolio balance over time

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        elif order.status in [order.Completed]:
            if order.isbuy():
                buy_price = order.executed.price
                self.running_portfolio_balance = self.broker.getvalue()
                self.buy_date = self.datas[0].datetime.date(0)  # Record the buy date
                self.buy_shares = round(order.executed.size)  # Round shares to the nearest whole number
                commission = buy_price * self.buy_shares * self.params.commission_rate
                print(f"BUY EXECUTED on {self.buy_date}: {buy_price}, Shares: {self.buy_shares}, Commission: {commission:.2f}")
                self.trades.append({
                    'Date': self.buy_date,
                    'Type': 'BUY',
                    'Price': round(buy_price, 2),
                    'Shares': self.buy_shares,
                    'Pct_Change': 0.0,
                    'Portfolio_Balance': f"{self.running_portfolio_balance:,.2f}",
                    'Duration': '',
                    'Commission': round(commission, 2)
                })
            elif order.issell():
                sell_price = order.executed.price
                last_trade = self.trades[-1]
                buy_price = last_trade['Price']
                pct_change = ((sell_price - buy_price) / buy_price) * 100
                self.running_portfolio_balance = self.broker.getvalue()
                sell_date = self.datas[0].datetime.date(0)
                commission = sell_price * self.buy_shares * self.params.commission_rate
                duration = self.calculate_trading_days(self.buy_date, sell_date)
                print(f"SELL EXECUTED on {sell_date}: {sell_price}, Pct Change: {pct_change:.2f}, Duration: {duration} trading days, Commission: {commission:.2f}")
                self.trades.append({
                    'Date': sell_date,
                    'Type': 'SELL',
                    'Price': round(sell_price, 2),
                    'Shares': self.buy_shares,
                    'Pct_Change': round(pct_change, 2),
                    'Portfolio_Balance': f"{self.running_portfolio_balance:,.2f}",
                    'Duration': duration,
                    'Commission': round(commission, 2)
                })
                self.buy_shares = 0  # Reset after selling
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order Failed: {order.status}")
            self.order = None

    def calculate_trading_days(self, start_date, end_date):
        """
        Calculate the number of trading days (weekdays) between two dates.
        Excludes weekends and assumes start_date < end_date.
        """
        current_date = start_date + pd.Timedelta(days=1)  # Start counting from the day after the buy date
        trading_days = 0

        print(f"\nCalculating trading days from {start_date} to {end_date}")

        while current_date <= end_date:
            print(f"Checking date: {current_date}")

            if current_date.weekday() < 5:  # Monday to Friday are 0-4
                print(f"Adding trading day: {current_date}")
                trading_days += 1
            else:
                print(f"Skipping weekend: {current_date}")
            
            current_date += pd.Timedelta(days=1)

        print(f"Total trading days between {start_date} and {end_date}: {trading_days}")
        return trading_days

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        print(f"\nDate: {current_date}")
        print(f"Open: {self.dataopen[0]:.2f}, Close: {self.dataclose[0]:.2f}")
        print(f"EMA Short: {self.ema_short[0]:.2f}, EMA Long: {self.ema_long[0]:.2f}")
        print(f"EMA Short (Prev): {self.ema_short[-1]:.2f}, EMA Long (Prev): {self.ema_long[-1]:.2f}")

        # Track the portfolio balance at the end of each trading day
        self.portfolio_value_history.append({
            'Date': current_date,
            'Portfolio_Balance': self.broker.getvalue()
        })

        if self.order:
            print("Order pending, skipping iteration.")
            return

        if not self.position:
            if self.ema_short[0] > self.ema_long[0] and self.ema_short[-1] <= self.ema_long[-1]:
                cash_available = self.broker.get_cash()
                stake = cash_available * self.params.stake_percentage
                size = round(stake / self.dataopen[1])  # Round to nearest whole number
                print(f"Placing buy order for {size} shares at the next open price.")
                self.order = self.buy(size=size, exectype=bt.Order.Market)

        if self.position:
            if self.ema_short[0] < self.ema_long[0] and self.ema_short[-1] >= self.ema_long[-1]:
                print(f"Placing sell order for {self.position.size:.2f} shares at the current close price.")
                self.order = self.sell(size=self.position.size, exectype=bt.Order.Market)

        if self.position:
            print(f"Current position size: {self.position.size:.2f}")
        else:
            print("No position held.")

    def stop(self):
        final_value = self.broker.getvalue()
        print(f'\nFinal Portfolio Value: {final_value:,.2f}')

        # Print the results and save to a CSV
        file_path = os.path.join(results_dir, 'Apple_Performance.csv')
        print("\nTrade Summary:")
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Type", "Price", "Shares", "Pct_Change", "Portfolio_Balance", "Duration", "Commission"])
            for trade in self.trades:
                pct_change_str = f"{trade['Pct_Change']:.2f}" if trade['Type'] == 'SELL' else ''
                print(f"Date: {trade['Date']}, Type: {trade['Type']}, Price: {trade['Price']:.2f}, Shares: {trade['Shares']}, "
                      f"Pct Change: {pct_change_str}, "
                      f"Portfolio Balance: {trade['Portfolio_Balance']}, Duration: {trade['Duration']}, "
                      f"Commission: {trade['Commission']:.2f}")
                writer.writerow([
                    trade['Date'], trade['Type'], trade['Price'], trade['Shares'],
                    pct_change_str, trade['Portfolio_Balance'], trade['Duration'], trade['Commission']
                ])
            writer.writerow(["Final Portfolio Value", "", "", "", "", f"{final_value:,.2f}", "", ""])

        # Save portfolio value history
        self.save_portfolio_history()

    def save_portfolio_history(self):
        df = pd.DataFrame(self.portfolio_value_history)
        df.to_csv(os.path.join(results_dir, 'Apple_Portfolio_History.csv'), index=False)

# Function to plot portfolio performance
def plot_portfolio():
    df = pd.read_csv(os.path.join(results_dir, 'Apple_Portfolio_History.csv'))
    df['Date'] = pd.to_datetime(df['Date'])

    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Portfolio_Balance'], label='Portfolio Balance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Balance ($)')
    plt.title('Portfolio Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'ema_crossover_portfolio_performance.png'))
    plt.show()

# Main function to run the backtest
if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(100000)

    # Add commission to the broker
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Load data for AAPL
    data = bt.feeds.YahooFinanceCSVData(dataname='/Users/ryangalitzdorfer/Downloads/Market Machine/Stock Filtration/Individual Stock Data/AAPL_Data.csv')
    cerebro.adddata(data)

    # Add strategy to Cerebro
    cerebro.addstrategy(EMACrossoverStrategy)

    # Run backtest
    cerebro.run()

    # Plot portfolio performance
    plot_portfolio()
