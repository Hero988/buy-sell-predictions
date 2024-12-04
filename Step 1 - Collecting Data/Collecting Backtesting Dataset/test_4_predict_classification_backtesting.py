import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
from dotenv import load_dotenv
import random
import time

# Load environment variables for MT5 login credentials
load_dotenv()
login = int(os.getenv('MT5_LOGIN'))  # Replace with your login ID
password = os.getenv('MT5_PASSWORD')  # Replace with your password
server = os.getenv('MT5_SERVER')  # Replace with your server name

# Initialize MetaTrader 5 connection
if not mt5.initialize(login=login, password=password, server=server):
    print("Failed to initialize MT5, error code:", mt5.last_error())
    quit()

# Step 1: Gather Forex Data
def gather_forex_data(symbol, timeframe, start_date, end_date):
    timeframes = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    if timeframe not in timeframes:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Valid options: {list(timeframes.keys())}")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    rates = mt5.copy_rates_range(symbol, timeframes[timeframe], start, end)

    if rates is None:
        print(f"No data retrieved for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# Step 3: Generate and Save Charts
def generate_and_save_chart(df, filename):
    """
    Generate a candlestick chart without axes and save it as an image.
    """
    custom_colors = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black')
    custom_style = mpf.make_mpf_style(marketcolors=custom_colors, base_mpf_style='classic')

    mpf.plot(
        df,
        type='candle',
        style=custom_style,
        savefig=dict(fname=filename, dpi=100),
        axisoff=True  # Removes axes and gridlines for cleaner visuals
    )

# Step 4: Save Charts and Data in Symbol-Specific Folders
def save_candlestick_data(data, sequence_length, future_period, output_dir, symbol):
    """
    Save historical data, future data, historical chart, future chart, and full chart
    in symbol-specific folders.
    """
    # Create the folder for the specific symbol
    symbol_folder = os.path.join(output_dir, symbol)
    os.makedirs(symbol_folder, exist_ok=True)

    for i in range(len(data) - sequence_length - future_period):
        # Get historical, future, and full data
        historical_data = data.iloc[i:i + sequence_length]
        future_data = data.iloc[i + sequence_length:i + sequence_length + future_period]
        combined_data = data.iloc[i:i + sequence_length + future_period]

        # Create a folder for the current sequence within the symbol folder
        sequence_folder = os.path.join(symbol_folder, f"{symbol}_sequence_{i}")
        os.makedirs(sequence_folder, exist_ok=True)

        # Save historical and future data as CSV
        historical_csv_path = os.path.join(sequence_folder, "historical_data.csv")
        future_csv_path = os.path.join(sequence_folder, "future_data.csv")
        historical_data.to_csv(historical_csv_path)
        future_data.to_csv(future_csv_path)

        # Save charts
        historical_chart_path = os.path.join(sequence_folder, "historical_chart.png")
        future_chart_path = os.path.join(sequence_folder, "future_chart.png")
        full_chart_path = os.path.join(sequence_folder, "full_chart.png")
        
        generate_and_save_chart(historical_data, historical_chart_path)
        generate_and_save_chart(future_data, future_chart_path)
        generate_and_save_chart(combined_data, full_chart_path)

# Step 5: Main Function
if __name__ == "__main__":
    symbols = [
        "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCAD",
        "AUDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
        "NZDUSD", "CHFJPY", "EURGBP", "EURAUD", "EURCHF",
        "EURJPY", "EURNZD", "EURCAD", "GBPCHF", "GBPJPY",
        "CADCHF", "CADJPY", "GBPAUD", "GBPCAD", "GBPNZD",
        "NZDCAD", "NZDCHF", "NZDJPY"
    ]
    today = datetime.today()
    start_date = datetime(2024, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    timeframe = "1h"
    output_dir = "candlestick_data_backtesting"
    sequence_length = 12
    future_period = 6

    for symbol in symbols:
        raw_data = gather_forex_data(symbol, timeframe, start_date, end_date)
        if raw_data is not None:
            print(f"Processing: {symbol} | Current Time {datetime.now()}")
            save_candlestick_data(raw_data, sequence_length, future_period, output_dir, symbol)