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

# Step 2: Calculate Indicators
def calculate_indicators(data):
    data['MA_5'] = data['close'].rolling(window=5).mean()
    data['MA_20'] = data['close'].rolling(window=20).mean()
    data['BB_upper'] = data['MA_20'] + 2 * data['close'].rolling(window=20).std()
    data['BB_lower'] = data['MA_20'] - 2 * data['close'].rolling(window=20).std()
    return data.dropna()

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
        axisoff=True  # Removes axes and gridlines for cleaner YOLO input
    )

# Step 4: Annotate and Save Bounding Boxes for YOLO
def save_annotations(data, class_label, annotation_path):
    """
    Save annotations in YOLO format.
    """
    with open(annotation_path, 'w') as f:
        for index, row in data.iterrows():
            # YOLO format requires normalized x_center, y_center, width, height
            x_center = (index - data.index[0]).total_seconds() / (data.index[-1] - data.index[0]).total_seconds()
            y_center = (row['high'] + row['low']) / 2  # Midpoint of high and low
            y_center_normalized = (y_center - data['low'].min()) / (data['high'].max() - data['low'].min())
            width = 1 / len(data)  # Assuming uniform spacing for candlesticks
            height = (row['high'] - row['low']) / (data['high'].max() - data['low'].min())

            # Write YOLO formatted annotation
            f.write(f"{class_label} {x_center:.6f} {y_center_normalized:.6f} {width:.6f} {height:.6f}\n")

# Step 5: Save Charts and Annotations in Class Folders
def save_candlestick_charts(data, sequence_length, future_period, output_dir, symbol):
    """
    Save historical, future, and full charts in class folders ('buy' or 'sell'),
    and save YOLO annotations.
    """
    for i in range(len(data) - sequence_length - future_period):
        # Get historical, future, and full data
        historical_data = data.iloc[i:i + sequence_length]
        future_data = data.iloc[i + sequence_length:i + sequence_length + future_period]
        combined_data = data.iloc[i:i + sequence_length + future_period]

        # Get the last close price from the historical data
        last_close = historical_data['close'].iloc[-1]

        # Classification based on future candlestick prices
        if (future_data['close'] > last_close).all() and (future_data['high'] > last_close).all() and (future_data['low'] > last_close).all():
            class_folder = "buy"
            class_label = 0  # YOLO class ID
        elif (future_data['close'] < last_close).all() and (future_data['high'] < last_close).all() and (future_data['low'] < last_close).all():
            class_folder = "sell"
            class_label = 1  # YOLO class ID
        else:
            continue  # Skip this sequence if neither condition is met

        # Create a folder for the current sequence within the class folder
        sequence_folder = os.path.join(output_dir, class_folder, f"{symbol}_sequence_{i}")
        os.makedirs(sequence_folder, exist_ok=True)

        # Save charts
        historical_path = os.path.join(sequence_folder, "historical_chart.png")
        future_path = os.path.join(sequence_folder, "future_chart.png")
        full_path = os.path.join(sequence_folder, "full_chart.png")
        annotation_path = os.path.join(sequence_folder, "annotations.txt")

        generate_and_save_chart(historical_data, historical_path)
        generate_and_save_chart(future_data, future_path)
        generate_and_save_chart(combined_data, full_path)

        # Save YOLO annotations
        save_annotations(historical_data, class_label, annotation_path)

# Step 6: Main Function
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
    start_date = (today - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
    end_date = datetime(2023, 12, 31).strftime("%Y-%m-%d")
    timeframe = "1h"
    output_dir = "candlestick_charts"
    sequence_length = 12
    future_period = 6

    for symbol in symbols:
        raw_data = gather_forex_data(symbol, timeframe, start_date, end_date)
        if raw_data is not None:
            print(f"Processing: {symbol} | Current Time {datetime.now()}")
            enhanced_data = calculate_indicators(raw_data)
            save_candlestick_charts(enhanced_data, sequence_length, future_period, output_dir, symbol)