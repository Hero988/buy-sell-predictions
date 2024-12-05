import numpy as np
import pandas as pd
import os
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sklearn.preprocessing import MinMaxScaler
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib
import mplfinance as mpf
from dotenv import load_dotenv
import random
import time
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, ViTForImageClassification, AutoImageProcessor, ConvNextForImageClassification, AutoModelForImageClassification, AutoConfig
from safetensors.torch import load_file
from PIL import Image
import torch
import kagglehub
import sys
import shutil

# Load environment variables for MT5 login credentials
load_dotenv()
login = int(os.getenv('MT5_LOGIN'))  # Replace with your login ID
password = os.getenv('MT5_PASSWORD')  # Replace with your password
server = os.getenv('MT5_SERVER')  # Replace with your server name

# Initialize MetaTrader 5 connection
if not mt5.initialize(login=login, password=password, server=server):
    print("Failed to initialize MT5, error code:", mt5.last_error())
    quit()

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

# Function to get prediction from the model
def predict_image(image_path, model, image_processor):
  """
  Predict the class and confidence for a given image.

  Args:
      image_path (str): Path to the image file.
      model (torch.nn.Module): The trained model.
      image_processor: The image processor for the model.

  Returns:
      tuple: Predicted class, confidence score.
  """
  # Load and preprocess the image
  image = Image.open(image_path).convert("RGB")
  inputs = image_processor(images=image, return_tensors="pt")

  # Perform inference
  with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits
      probabilities = torch.nn.functional.softmax(logits, dim=-1)
      confidence, predicted_class = torch.max(probabilities, dim=-1)

  # Convert class and confidence to values
  predicted_class = predicted_class.item()
  confidence = confidence.item()
  return predicted_class, confidence, probabilities

def get_most_recent_candles(symbol, timeframe="1h", num_candles=12):
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

    # Get the last `num_candles` bars of data, excluding the most recent candle
    start_pos = 1  # Skip the most recent candle
    rates = mt5.copy_rates_from_pos(symbol, timeframes[timeframe], start_pos, num_candles)

    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def download_and_extract_model(filename="output_latest", filename_output="output"):
    """
    Downloads, extracts, and deletes a zip file if it doesn't already exist.

    Args:
        filename: The name of the zip file to download.
        filename_output: The name of the folder to extract the contents to.
    """
    if not os.path.exists(filename_output):
        try:
            # Download the .zip file
            print(f"Downloading '{filename}'...")
            # Get the actual downloaded file path
            downloaded_file_path = kagglehub.model_download("sulimantadros/hugging_face_model_facebookconvnext-base-224_64/other/default")

            # Rename if necessary (if kagglehub returns a different name)
            if downloaded_file_path != filename:
                os.rename(downloaded_file_path, filename)
                print(f"Renamed downloaded file to '{filename}'")

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"File '{filename_output}' already exists. Skipping download and extraction.")

def execute_trade(symbol, predicted_label, volume):
    """
    Executes a buy or sell order based on the predicted label, with stop loss and take profit.
    Args:
        symbol (str): The symbol to trade.
        predicted_label (str): 'buy' or 'sell'.
        volume (float): The volume of the trade (lot size).
    """
    # Determine action type
    action = mt5.ORDER_TYPE_BUY if predicted_label == "buy" else mt5.ORDER_TYPE_SELL

    # Get the latest price
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info is None:
        print(f"Could not retrieve tick info for {symbol}. Trade aborted.")
        return

    price = tick_info.ask if action == mt5.ORDER_TYPE_BUY else tick_info.bid
    point = mt5.symbol_info(symbol).point  # Symbol point size
    deviation = 20  # Allowed slippage in points

    # Calculate SL and TP
    #sl = price - 100 * point if action == mt5.ORDER_TYPE_BUY else price + 100 * point
    #tp = price + 200 * point if action == mt5.ORDER_TYPE_BUY else price - 200 * point

    # Prepare the trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": action,
        "price": price,
        #"sl": sl,
        #"tp": tp,
        "deviation": deviation,
        "magic": 234000,  # Custom identifier for the trade
        "comment": "Trade executed by script",
        "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or cancel
    }

    # Send the trade request
    result = mt5.order_send(request)
    if result is None:
        print(f"Failed to send trade request for {symbol}.")
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to execute {predicted_label} order for {symbol}: {result.retcode}")
        print(f"Error details: {mt5.last_error()}")

def close_all_positions():
    """
    Closes all open positions on the account using their tickets.
    """
    # Retrieve all open positions
    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        print("No open positions to close.")
        return

    for position in positions:
        symbol = position.symbol
        volume = position.volume
        position_type = position.type  # 0 = Buy, 1 = Sell

        # Determine the opposite action for closing the position
        action = mt5.ORDER_TYPE_BUY if position_type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL

        # Get the current price for the action
        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            print(f"Failed to retrieve tick info for {symbol}. Skipping position closure.")
            continue

        # Select the appropriate price
        price = tick_info.ask if action == mt5.ORDER_TYPE_BUY else tick_info.bid
        deviation = 20  # Allowed slippage in points

        print(f"  Volume: {volume}, Type: {'Buy' if position_type == mt5.ORDER_TYPE_BUY else 'Sell'}")
        print(f"  Opposite Action: {'Buy' if action == mt5.ORDER_TYPE_BUY else 'Sell'}, Price: {price}, Deviation: {deviation}")

        # Create a close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
            "price": tick_info.ask if position.type == 1 else tick_info.bid,
            "deviation": 20,
            "magic": 234000,
            "comment": f"python_script_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send the close request
        result = mt5.order_send(request)
        if result is None:
            print(f"Failed to send close order for position ticket {position.ticket}.")
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close position ticket {position.ticket} for {symbol}: {result.retcode}")
            print(f"Error details: {mt5.last_error()}")

# Main function
if __name__ == "__main__":
    symbols = [
        "EURNZD", "GBPCAD", "GBPNZD", "AUDCAD", "GBPUSD", 
        #"AUDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
        #"NZDUSD", "CHFJPY", "EURGBP", "EURAUD", "EURCHF",
        #"EURJPY", , "EURCAD", "GBPCHF", "GBPJPY",
        #"CADCHF", "CADJPY", "GBPAUD", "USDCHF", "USDCAD",
        #"NZDCAD", "NZDCHF", "NZDJPY", "USDJPY"
    ]

    # USDCAD, GBPUSD, GBPCAD, EURNZD, AUDCAD best so far

    id2label = {0: "buy", 1: "sell"}

    # Load the model and image processor
    model_name = "facebook/convnext-base-224"
    config = AutoConfig.from_pretrained(model_name, num_labels=2, id2label={0: "buy", 1: "sell"}, label2id={"buy": 0, "sell": 1})

    model = AutoModelForImageClassification.from_config(config)
    model.eval()
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Check and download the model weights
    url_for_model = "sulimantadros/hugging_face_model_facebookconvnext-base-224_64/other/default"
    model_weights_path = "output_latest_model.zip/saved_model_1_no_threshold_based_model_64%/model.safetensors"

    if not os.path.exists(model_weights_path):
        print("Model weights not found. Downloading...")
        download_and_extract_model(
            filename="output_latest_model.zip",
            filename_output="output_latest_model",
            url=url_for_model
        )
    else:
        print("Model weights already exist. Skipping download.")

    # Load model weights
    try:
        weights = load_file(model_weights_path)
        model.load_state_dict(weights)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    trade_results_folder = 'trade_results'
    os.makedirs(trade_results_folder, exist_ok=True)

    while True:
        print(f"Starting new loop at {datetime.now()}")
        for symbol in symbols:
            recent_candles = get_most_recent_candles(symbol, "1h", num_candles=12)
            if recent_candles is not None:
                recent_candles['time'] = pd.to_datetime(recent_candles['time'])
                recent_candles.set_index('time', inplace=True)
                latest_time = recent_candles.index[-1]

                # Create directories for the symbol
                symbol_folder = os.path.join(trade_results_folder, symbol)

                # Generate and save chart
                chart_filename = f"{symbol}.png"
                generate_and_save_chart(recent_candles, chart_filename)

                # Predict
                predicted_class, confidence, probabilities = predict_image(chart_filename, model, image_processor)
                predicted_label = id2label.get(predicted_class, "Unknown")

                # Format the confidence to 2 decimal places for clarity
                formatted_confidence = f"{confidence:.2f}"

                # Create the new filename with the confidence level
                new_filename = f"{symbol}_{predicted_label}_{formatted_confidence}_{predicted_class}.png"

                # Rename the file
                os.rename(chart_filename, new_filename)

                if confidence > 0.9:
                    print(f"{confidence} > 0.9 making trade at {datetime.now()} for symbol {symbol}")
                    execute_trade(symbol, predicted_label, volume=0.2)
                    os.makedirs(symbol_folder, exist_ok=True)
                    # Create subfolder structure with the date and hour of day
                    date_folder = latest_time.strftime('%Y-%m-%d')  # e.g., '2024-12-02'
                    hour_of_day = latest_time.strftime('%H')  # 24-hour format, e.g., '14' for 2 PM

                    # Define the path for the folder with the new structure
                    data_folder = os.path.join(symbol_folder, "data", date_folder, hour_of_day)

                    # Create the directories if they don't already exist
                    os.makedirs(data_folder, exist_ok=True)

                    # Save the recent candles to a CSV file in the same folder
                    candles_csv_path = os.path.join(data_folder, f"{symbol}_candles.csv")
                    recent_candles.to_csv(candles_csv_path, index=True)

                    # Append data to the CSV
                    csv_path = os.path.join(symbol_folder, "trade_data.csv")
                    new_data = {
                        "confidence": [confidence],
                        "probabilities": [probabilities],
                        "predicted_label": [predicted_label],
                        "latest_time": [latest_time]
                    }
                    df = pd.DataFrame(new_data)
                    if os.path.exists(csv_path):
                        df.to_csv(csv_path, mode='a', header=False, index=False)
                    else:
                        df.to_csv(csv_path, index=False)
                    # Move the file to the destination directory
                    shutil.move(new_filename, data_folder)
                else:
                    print(f"{confidence} < 0.9 skipping symbol {symbol} with predicted label {predicted_label}")
                    os.remove(new_filename)

        # Add 6 hours to the latest time
        countdown_time = 6 * 60 * 60  # 6 hours in seconds
        print("Sleeping for 6 hours...")
        while countdown_time > 0:
            hours, remainder = divmod(countdown_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            timer = f"{hours:02}:{minutes:02}:{seconds:02}"
            print(f"Time remaining until next iteration: {timer}", end="\r")
            time.sleep(1)
            countdown_time -= 1
        # Close all open positions
        close_all_positions()
        print("\nRestarting loop...")
            
