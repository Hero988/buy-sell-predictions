import os
import random
import shutil
import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoImageProcessor,
)
from datasets import load_dataset
import kagglehub
from safetensors.torch import load_file
import MetaTrader5 as mt5
from dotenv import load_dotenv
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

# Utility Functions
def download_and_extract_data_predictions(filename, filename_output, url):
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
          downloaded_file_path = kagglehub.dataset_download(url)

          # Rename if necessary (if kagglehub returns a different name)
          if downloaded_file_path != filename:
              os.rename(downloaded_file_path, filename)
              print(f"Renamed downloaded file to '{filename}'")

      except Exception as e:
          print(f"An error occurred: {e}")
  else:
      print(f"File '{filename_output}' already exists. Skipping download and extraction.")

def download_and_extract_model(filename, filename_output, url):
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
            downloaded_file_path = kagglehub.model_download(url)

            # Rename if necessary (if kagglehub returns a different name)
            if downloaded_file_path != filename:
                os.rename(downloaded_file_path, filename)
                print(f"Renamed downloaded file to '{filename}'")

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"File '{filename_output}' already exists. Skipping download and extraction.")

# Split and Rename Data
def split_and_rename_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    classes = ["buy", "sell"]

    for class_name in classes:
        class_dir = os.path.join(input_dir, class_name)
        subfolders = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))]
        random.shuffle(subfolders)
        train_count = int(len(subfolders) * train_ratio)
        val_count = int(len(subfolders) * val_ratio)

        splits = {
            "train": subfolders[:train_count],
            "validation": subfolders[train_count : train_count + val_count],
            "test": subfolders[train_count + val_count :],
        }

        for split, folders in splits.items():
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for folder in folders:
                folder_name = os.path.basename(folder)
                chart_path = os.path.join(folder, "historical_chart.png")
                if os.path.exists(chart_path):
                    shutil.copy(chart_path, os.path.join(split_class_dir, f"{folder_name}.png"))
                else:
                    print(f"Warning: {chart_path} not found in {folder}")
    print("Data split and renamed successfully.")

# Prediction and Evaluation Functions
def predict_image(image_path, model, image_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)
    return predicted_class.item(), confidence.item()

def process_folders(base_dir, model, image_processor, output_csv_path):
    predictions_dict = {}
    total_folders = sum([len(dirs) for _, dirs, _ in os.walk(base_dir)])
    completed_folders = 0

    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            chart_path = os.path.join(folder_path, "historical_chart.png")
            if os.path.exists(chart_path):
                predicted_class, confidence = predict_image(chart_path, model, image_processor)
                predictions_dict[folder_path] = {"predicted_class": predicted_class, "confidence": confidence}
            completed_folders += 1
            sys.stdout.write(f"\rProgress: {completed_folders}/{total_folders} folders completed")
            sys.stdout.flush()

    print("\nProcessing complete!")
    with open(output_csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Folder", "Predicted Class", "Confidence"])
        for folder, result in predictions_dict.items():
            writer.writerow([folder, result["predicted_class"], result["confidence"]])

# Profit and Drawdown Calculations
def calculate_profits_and_drawdowns(sorted_df, lot_size, starting_balance, output_folder, symbol):
    balance = starting_balance
    profits = []
    dates = []
    predicted_labels = []
    predictions_bool_list = []
    actual_labels = []
    folders = []
    id2label = {0: "buy", 1: "sell"}

    for folder in sorted_df["Folder"]:
        hist_csv = os.path.join(folder, "historical_data.csv")
        future_csv = os.path.join(folder, "future_data.csv")

        if not os.path.exists(hist_csv) or not os.path.exists(future_csv):
            print(f"Missing data in folder: {folder}")
            continue

        hist_data = pd.read_csv(hist_csv)
        future_data = pd.read_csv(future_csv)

        last_close_hist = hist_data["close"].iloc[-1]
        last_close_future = future_data["close"].iloc[-1]
        pred_class = sorted_df.loc[sorted_df["Folder"] == folder, "Predicted Class"].values[0]
        pred_label = id2label[pred_class]

        # Set stop loss and take profit thresholds
        stop_loss_limit = -100  # Maximum loss in monetary value
        take_profit_limit = 200  # Maximum profit in monetary value

        # Get symbol information and point size
        point = mt5.symbol_info(symbol).point  # Symbol point size

        # Initialize profit and trade status
        profit = 0
        trade_closed = False

        # Calculate SL and TP based on action
        sl = last_close_hist - 100 * point if pred_label == "buy" else last_close_hist + 100 * point
        tp = last_close_hist + 200 * point if pred_label == "buy" else last_close_hist - 200 * point

        # Loop through the future data to check high and low prices
        for index, row in future_data.iterrows():
            high_price = row["high"]
            low_price = row["low"]

            # Check for stop loss or take profit hit
            if pred_label == "buy":
                if low_price <= sl:  # Stop loss for buy
                    profit = stop_loss_limit
                    trade_closed = True
                    break
                elif high_price >= tp:  # Take profit for buy
                    profit = take_profit_limit
                    trade_closed = True
                    break
            elif pred_label == "sell":
                if high_price >= sl:  # Stop loss for sell
                    profit = stop_loss_limit
                    trade_closed = True
                    break
                elif low_price <= tp:  # Take profit for sell
                    profit = take_profit_limit
                    trade_closed = True
                    break

        # If the trade did not hit SL or TP, calculate profit based on final close price
        if not trade_closed:
            if pred_label == "buy":
                profit = (last_close_future - last_close_hist) * lot_size  # Profit for buy
            elif pred_label == "sell":
                profit = (last_close_hist - last_close_future) * lot_size  # Profit for sell

        # Update balance and record trade details
        balance += profit
        profits.append(balance)
        dates.append(future_data["time"].iloc[-1])
        predicted_labels.append(pred_label)
        folders.append(folder)
        actual_labels.append("buy" if last_close_future > last_close_hist else "sell")
        predictions_bool_list.append((pred_label == "buy" and last_close_future > last_close_hist) or (pred_label == "sell" and last_close_future < last_close_hist))

    # Create results DataFrame
    results_df = pd.DataFrame({
        "Date": dates,
        "Account Balance": profits,
        "Predicted Label": predicted_labels,
        "Actual Label": actual_labels,
        "Folder": folders,
        "Prediction Correct": predictions_bool_list,
    })

    # Calculate drawdowns
    results_df["Drawdown"] = results_df["Account Balance"].cummax() - results_df["Account Balance"]
    results_df.to_csv(os.path.join(output_folder, "trade_results.csv"), index=False)

    # Plot account balance over time
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(results_df["Date"]), results_df["Account Balance"], marker="o")
    plt.title("Account Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Account Balance")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "account_balance_plot.png"))

    return results_df

# Main Execution
if __name__ == "__main__":
    # Example of loading and evaluating the model
    model_name = "facebook/convnext-base-224"
    config = AutoConfig.from_pretrained(model_name, num_labels=2, id2label={0: "buy", 1: "sell"}, label2id={"buy": 0, "sell": 1})
    model = AutoModelForImageClassification.from_config(config)
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

    # Check and download the dataset
    url_for_data = "sulimantadros/partly-multiple-pairs-1-hour"
    dataset_path = "output_latest_data_part_multiple.zip"

    if not os.path.exists(dataset_path):
        print("Dataset not found. Downloading...")
        download_and_extract_data_predictions(
            filename="output_latest_data_part_multiple.zip",
            filename_output="output_latest_data_part_multiple",
            url=url_for_data
        )
    else:
        print("Dataset already exists. Skipping download.")


    # Define the base directory
    base_directory = "output_latest_data_part_multiple.zip/Pairs"

    # Get the list of folder names
    folder_names = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    output_folder_main = "output_backtesting_latest"
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder_main, exist_ok=True)

    starting_balance = 10000
    lot_size = 100000

    symbols = [
        "EURUSD", #"GBPUSD", "USDCHF", "USDJPY", "USDCAD",
        #"AUDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
        #"NZDUSD", "CHFJPY", "EURGBP", "EURAUD", "EURCHF",
        #"EURJPY", "EURNZD", "EURCAD", "GBPCHF", "GBPJPY",
        #"CADCHF", "CADJPY", "GBPAUD", "GBPCAD", "GBPNZD",
        #"NZDCAD", "NZDCHF", "NZDJPY"
    ]

    for symbol in symbols:
        print(f'Currently on Pair: {symbol}')
        # Example usage
        base_dir = f"{base_directory}/{symbol}"

        new_folder_path = os.path.join(output_folder_main, symbol)
        os.makedirs(new_folder_path, exist_ok=True)

        predictions_csv_path = os.path.join(new_folder_path, "predictions_results.csv")
        sorted_csv_path = os.path.join(new_folder_path, "sorted_filtered_predictions_results.csv")

        # Process folders and save predictions
        process_folders(base_dir, model, image_processor, predictions_csv_path)

        # Load and filter the results
        df = pd.read_csv(predictions_csv_path)
        threshold = 0.9
        filtered_df = df[df['Confidence'] > threshold].copy()
        filtered_df['Sequence_Number'] = filtered_df['Folder'].str.extract(r'sequence_(\d+)').astype(int)
        sorted_df = filtered_df.sort_values(by='Sequence_Number').drop(columns=['Sequence_Number'])
        sorted_df.to_csv(sorted_csv_path, index=False)

        # Calculate profits and drawdowns
        results_df = calculate_profits_and_drawdowns(sorted_df, lot_size, starting_balance, new_folder_path, symbol)

        # Visualize a random sequence
        #random_folder = random.choice(sorted_df['Folder'].tolist())
        #visualize_sequence(random_folder, model, image_processor, {0: "buy", 1: "sell"}, lot_size, new_folder_path)
