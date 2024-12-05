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
from backtesting_12_candles import download_and_extract_model

import os
import shutil
import pandas as pd

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
        #point = mt5.symbol_info(symbol).point  # Symbol point size

        # Initialize profit and trade status
        profit = 0
        trade_closed = False

        # Calculate SL and TP based on action
        #sl = last_close_hist - 100 * point if pred_label == "buy" else last_close_hist + 100 * point
        #tp = last_close_hist + 200 * point if pred_label == "buy" else last_close_hist - 200 * point

        """

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
        """

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

# Correctly get the file path for the nested pair folders
def get_pair_filepath(pair_name, base_dir):
    # Construct the nested folder path
    nested_folder_path = os.path.join(base_dir, pair_name, pair_name)
    # Check if the nested path exists; if not, return the top-level path
    if os.path.isdir(nested_folder_path):
        return nested_folder_path
    else:
        return os.path.join(base_dir, pair_name)


# Function to predict using the model
def predict_image(image_path, model, image_processor):
    try:
        image = Image.open(image_path).convert("RGB")  # Try to open the image
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)
        return predicted_class.item(), confidence.item()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None  # Return None values if prediction fails

# Function to process folders and generate predictions
def process_folders(base_dir, model, image_processor, output_csv_path):
    predictions_dict = {}
    total_folders = sum([len(dirs) for _, dirs, _ in os.walk(base_dir)])
    completed_folders = 0

    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            chart_path = os.path.join(folder_path, "historical_chart.png")
            
            # Check if the chart file exists
            if os.path.exists(chart_path):
                predicted_class, confidence = predict_image(chart_path, model, image_processor)
                
                # Log predictions only if valid
                if predicted_class is not None and confidence is not None:
                    predictions_dict[folder_path] = {"predicted_class": predicted_class, "confidence": confidence}
            
            completed_folders += 1
            sys.stdout.write(f"\rProgress: {completed_folders}/{total_folders} folders completed")
            sys.stdout.flush()

    print("\nProcessing complete!")
    # Save predictions to a CSV file
    with open(output_csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Folder", "Predicted Class", "Confidence"])
        for folder, result in predictions_dict.items():
            writer.writerow([folder, result["predicted_class"], result["confidence"]])
    print(f"Predictions saved to {output_csv_path}")

# Define the base directory for the output folders
output_directory = "output_backtesting_latest"

# Define the folder to store all combined contents in the current working directory
combined_folder_path = os.path.abspath("top_5_folderpaths_in_one/Main_Folder")
if not os.path.exists(combined_folder_path):
    os.makedirs(combined_folder_path)
else:
    print(f"Folder already exists: {combined_folder_path}")

def move_subfolders_to_combined(main_folder):
    # Define the destination folder
    combined_folder_path = os.path.join(main_folder, "backtest_combined")

    # Check if the combined folder already exists
    if os.path.exists(combined_folder_path):
        print(f"The folder '{combined_folder_path}' already exists. Skipping this step.")
        return  # Exit the function if the folder already exists

    # Create the combined folder
    os.makedirs(combined_folder_path, exist_ok=True)
    print(f"Created folder: {combined_folder_path}")

    # Move all subfolders in main_folder to the combined folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path) and subfolder != "backtest_combined":  # Avoid moving the combined folder itself
            dest_path = os.path.join(combined_folder_path, subfolder)
            try:
                shutil.move(subfolder_path, dest_path)
                print(f"Moved folder: {subfolder_path} to {dest_path}")
            except Exception as e:
                print(f"Failed to move folder {subfolder_path}. Error: {e}")

    print(f"All subfolders have been moved to '{combined_folder_path}'.")

# Initialize an empty list to store account balances and pairs
data = []

if not os.path.exists(combined_folder_path):
    # Walk through each folder in the output directory
    for pair_folder in os.listdir(output_directory):
        pair_folder_path = os.path.join(output_directory, pair_folder)
        if os.path.isdir(pair_folder_path):  # Ensure it's a folder
            # Define the path to the trade_results.csv file in the nested folder
            nested_trade_results_file = os.path.join(pair_folder_path, pair_folder, "trade_results.csv")
            top_level_trade_results_file = os.path.join(pair_folder_path, "trade_results.csv")
            
            # Check if the file exists in the nested folder first, then top-level folder
            if os.path.exists(nested_trade_results_file):
                trade_results_file = nested_trade_results_file
            elif os.path.exists(top_level_trade_results_file):
                trade_results_file = top_level_trade_results_file
            else:
                print(f"trade_results.csv not found in {pair_folder_path} or its nested folder.")
                continue
            
            # Read the trade_result.csv file into a dataframe
            df = pd.read_csv(trade_results_file)
            
            if not df.empty:  # Ensure the dataframe is not empty
                # Get the last recorded balance
                last_balance = df.iloc[-1]['Account Balance']
                
                # Append the data (Pair and Account Balance) to the list
                data.append({'Pair': pair_folder, 'Account Balance': last_balance})
            else:
                print(f"No data in {trade_results_file}")

    # Create a new dataframe from the collected data
    result_df = pd.DataFrame(data)

    # Remove rows where the Pair contains 'JPY'
    result_df = result_df[~result_df['Pair'].str.contains('JPY', case=False, na=False)]

    # Sort the dataframe by Account Balance in descending order
    result_df = result_df.sort_values(by='Account Balance', ascending=False)

    # Keep only the top 5 rows
    top_5_pairs = result_df.head(5)

    # Add the correct file path for each pair
    top_5_pairs['File Path'] = top_5_pairs['Pair'].apply(lambda pair: get_pair_filepath(pair, output_directory))

    # Combine all subfolders into one Main_Folder inside top_5_folderpaths_in_one
    for _, row in top_5_pairs.iterrows():
        pair_folder_path = row['File Path']
        
        # Get all nested folders inside the pair's folder
        if os.path.isdir(pair_folder_path):
            for subfolder in os.listdir(pair_folder_path):
                subfolder_path = os.path.join(pair_folder_path, subfolder)
                
                # Copy the subfolder into the Main_Folder
                if os.path.isdir(subfolder_path):
                    dest_path = os.path.join(combined_folder_path, os.path.basename(subfolder_path))
                    if not os.path.exists(dest_path):  # Copy only if it doesn't already exist
                        shutil.copytree(subfolder_path, dest_path, dirs_exist_ok=True)

    # Save the top 5 pairs dataframe to a CSV file
    result_csv_path = os.path.abspath("summary_account_balances.csv")
    top_5_pairs.to_csv(result_csv_path, index=False)
else:
    print('Folder Already Exists')

# Example usage
main_folder_path = "top_5_folderpaths_in_one\Main_Folder"  # Main folder path

# Move subfolders into backtest_combined
move_subfolders_to_combined(main_folder_path)

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

# Define the base directory
base_directory = 'top_5_folderpaths_in_one'

# Get the list of folder names
folder_names = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

output_folder_main = "output_backtesting_latest"
# Create the output directory if it doesn't exist
os.makedirs(output_folder_main, exist_ok=True)

starting_balance = 10000
lot_size = 20000 

# Example usage
for symbol in folder_names:
    print(f'Currently on Pair: {symbol}')
    
    base_dir = f"{base_directory}/{symbol}"  # Source folder
    new_folder_path = os.path.join(output_folder_main, symbol)  # Destination folder

    # Check if the folder already exists in the output folder
    if os.path.exists(new_folder_path):
        print(f"The folder for {symbol} already exists in {output_folder_main}. Moving to the next symbol.")

    # Create the destination folder if it doesn't exist
    os.makedirs(new_folder_path, exist_ok=True)

    # Create a subfolder with the name of the symbol inside the new_folder_path
    symbol_subfolder_path = os.path.join(new_folder_path, symbol)
    os.makedirs(symbol_subfolder_path, exist_ok=True)

    #predictions_csv_path = os.path.join(new_folder_path, "predictions_results.csv")
    #sorted_csv_path = os.path.join(new_folder_path, "sorted_filtered_predictions_results.csv")

    # Process folders and save predictions
    #process_folders(base_dir, model, image_processor, predictions_csv_path)

    # Load and filter the results
    #df = pd.read_csv(predictions_csv_path)
    #threshold = 0.9
    #filtered_df = df[df['Confidence'] > threshold].copy()
    #filtered_df['Sequence_Number'] = filtered_df['Folder'].str.extract(r'sequence_(\d+)').astype(int)
    #sorted_df = filtered_df.sort_values(by='Sequence_Number').drop(columns=['Sequence_Number'])
    #sorted_df.to_csv(sorted_csv_path, index=False)

    sorted_df = pd.read_csv('output_backtesting_latest\Main_Folder\sorted_filtered_predictions_results.csv')

    # Calculate profits and drawdowns
    results_df = calculate_profits_and_drawdowns(sorted_df, lot_size, starting_balance, new_folder_path, symbol)

    # Visualize a random sequence
    #random_folder = random.choice(sorted_df['Folder'].tolist())
    #visualize_sequence(random_folder, model, image_processor, {0: "buy", 1: "sell"}, lot_size, new_folder_path)