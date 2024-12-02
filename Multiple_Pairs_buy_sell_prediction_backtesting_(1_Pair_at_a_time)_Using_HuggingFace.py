#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[6]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
import joblib
from PIL import Image
import tensorflow as tf
import re
import random
import tf_keras
from tf_keras.callbacks import EarlyStopping  # Use tf.keras.callbacks
import kagglehub
import zipfile
from transformers import TFAutoModel
import shutil
from zipfile import ZipFile
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, ViTForImageClassification, AutoImageProcessor, ConvNextForImageClassification, AutoModelForImageClassification, AutoConfig
import torch
from torch.utils.data import DataLoader
try:
  from datasets import load_dataset
except:
  get_ipython().system('pip install datasets')
  from datasets import load_dataset
try:
  import evaluate
except:
  get_ipython().system('pip install evaluate')
  import evaluate
from sklearn.metrics import classification_report
from safetensors.torch import load_file
import csv
import sys
import time


# ## Downloading the Data (testing data)

# In[7]:


def download_and_extract_data(filename="output_latest", filename_output="output"):
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
            downloaded_file_path = kagglehub.dataset_download("sulimantadros/buy-sell-object-detection-dataset")

            # Rename if necessary (if kagglehub returns a different name)
            if downloaded_file_path != filename:
                os.rename(downloaded_file_path, filename)
                print(f"Renamed downloaded file to '{filename}'")

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"File '{filename_output}' already exists. Skipping download and extraction.")

download_and_extract_data(filename="output_latest_data.zip", filename_output="output")


# ## Downloading the model

# In[8]:


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

download_and_extract_model(filename="output_latest_model.zip", filename_output="output")


# ## Sorting out the train, test and evaluate split

# In[9]:


def split_and_rename_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits and renames data from `input_dir` into train, validation, and test folders.

    Parameters:
    - input_dir: Path to the input directory containing `buy` and `sell` folders.
    - output_dir: Path to the output directory where the split data will be saved.
    - train_ratio: Proportion of data for training.
    - val_ratio: Proportion of data for validation.
    - test_ratio: Proportion of data for testing.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Define subdirectories for buy and sell
    classes = ['buy', 'sell']

    for class_name in classes:
        class_dir = os.path.join(input_dir, class_name)

        # Get all subfolders in the class directory
        subfolders = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))]

        # Shuffle subfolders randomly
        random.shuffle(subfolders)

        # Compute split indices
        train_count = int(len(subfolders) * train_ratio)
        val_count = int(len(subfolders) * val_ratio)

        train_folders = subfolders[:train_count]
        val_folders = subfolders[train_count:train_count + val_count]
        test_folders = subfolders[train_count + val_count:]

        # Create output subdirectories
        for split, split_folders in zip(['train', 'validation', 'test'], [train_folders, val_folders, test_folders]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            # Process each folder in the split
            for folder in split_folders:
                folder_name = os.path.basename(folder)  # Get the folder name
                historical_chart_path = os.path.join(folder, "historical_chart.png")

                if os.path.exists(historical_chart_path):
                    # Rename the file to <foldername>.png
                    new_file_name = f"{folder_name}.png"
                    new_file_path = os.path.join(split_class_dir, new_file_name)

                    # Copy the file to the destination
                    shutil.copy(historical_chart_path, new_file_path)
                else:
                    print(f"Warning: {historical_chart_path} not found in {folder}")

    print(f"Data successfully split into {output_dir}/train, {output_dir}/validation, and {output_dir}/test")


# In[10]:


# Example usage
input_dir = '/content/output_latest_data.zip/candlestick_charts'  # Path to the input directory with `buy` and `sell` folders
output_dir = '/content/output_folder'  # Path to the output directory
split_and_rename_data(input_dir, output_dir)


# ## All Functions Needed for Evaluation and Predictions

# In[11]:


# Define the dataset and preprocessing function
def preprocess_images(examples):
  """
  Preprocess images for evaluation.
  """
  images = [
      Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path
      for image_path in examples['image_path']
  ]
  inputs = image_processor(images=images, return_tensors="pt")
  inputs['labels'] = torch.tensor(examples['label'])
  return inputs


# In[12]:


def evaluate_model_with_threshold_report(model, dataloader, id2label, threshold):
  """
  Evaluate the model on the test set with a confidence threshold and generate classification reports.

  Args:
      model (torch.nn.Module): The trained model.
      dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
      id2label (dict): Mapping from class IDs to class labels.
      threshold (float): Confidence threshold to filter predictions.

  Returns:
      dict: Overall classification report.
      dict: Filtered classification report (confidence > threshold).
  """
  model.eval()
  y_true = []
  y_pred = []
  filtered_y_true = []
  filtered_y_pred = []

  with torch.no_grad():
      for batch in dataloader:
          inputs = batch['pixel_values']
          labels = batch['labels']

          # Forward pass
          outputs = model(pixel_values=inputs)
          logits = outputs.logits
          probabilities = torch.nn.functional.softmax(logits, dim=-1)
          confidence, predicted_class = torch.max(probabilities, dim=1)

          # Collect all predictions
          y_true.extend(labels.cpu().numpy())
          y_pred.extend(predicted_class.cpu().numpy())

          # Filter predictions based on the threshold
          for i in range(len(confidence)):
              if confidence[i] > threshold:
                  filtered_y_true.append(labels[i].item())
                  filtered_y_pred.append(predicted_class[i].item())

  # Generate overall classification report
  overall_report = classification_report(
      y_true,
      y_pred,
      target_names=[id2label[i] for i in range(len(id2label))],
      output_dict=True
  )
  print("\nOverall Classification Report:")
  print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))]))

  # Generate filtered classification report
  if filtered_y_true and filtered_y_pred:
      filtered_report = classification_report(
          filtered_y_true,
          filtered_y_pred,
          target_names=[id2label[i] for i in range(len(id2label))],
          output_dict=True
      )
      print(f"\nFiltered Classification Report (Confidence > {threshold}):")
      print(classification_report(filtered_y_true, filtered_y_pred, target_names=[id2label[i] for i in range(len(id2label))]))
  else:
      filtered_report = {}
      print(f"\nNo predictions with confidence > {threshold}. Try lowering the threshold.")

  # y_pred for overall
  y_pred_length_overall_results = len(y_pred)

  y_pred_length_filtered_results = len(filtered_y_pred)

  return overall_report, filtered_report, y_pred_length_overall_results, y_pred_length_filtered_results


# In[13]:


# Define a collate function for DataLoader
def collate_fn(batch):
  pixel_values = torch.stack([x['pixel_values'] for x in batch])
  labels = torch.tensor([x['labels'] for x in batch])
  return {'pixel_values': pixel_values, 'labels': labels}


# In[14]:


def predict(image_path):
  """
  Predict the class of an image using the loaded model.
  Args:
      image_path (str): Path to the input image.
  Returns:
      dict: Predicted class and confidence score.
  """
  # Open the image
  image = Image.open(image_path).convert("RGB")

  # Preprocess the image
  inputs = image_processor(images=image, return_tensors="pt")

  # Perform inference
  with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits
      probabilities = torch.nn.functional.softmax(logits, dim=-1)
      predicted_class_idx = torch.argmax(probabilities).item()
      predicted_class = id2label[predicted_class_idx]
      confidence = probabilities[0, predicted_class_idx].item()

  return {"class": predicted_class, "confidence": confidence}


# In[15]:


def visualize_prediction(threshold):
  """
  Keep selecting a random test image until a prediction with confidence above the threshold is found,
  then display the results.

  Parameters:
      threshold (float): Minimum confidence score required to visualize the prediction.
  """
  while True:
      # Choose a random image from test folder
      random_class = random.choice(['buy', 'sell'])
      class_folder = os.path.join(test_image_base_path, random_class)
      random_image_name = random.choice(os.listdir(class_folder))
      test_image_path = os.path.join(class_folder, random_image_name)

      # Perform prediction
      prediction = predict(test_image_path)
      predicted_class = prediction['class']
      confidence = prediction['confidence']

      # Check if confidence meets the threshold
      if confidence >= threshold:
          break  # Exit loop if a valid prediction is found

  # Extract folder name from the image name
  folder_name = os.path.splitext(os.path.basename(test_image_path))[0]

  # Find the corresponding folder in candlestick charts
  corresponding_folder = os.path.join(candlestick_charts_path, random_class, folder_name)
  full_chart_path = os.path.join(corresponding_folder, 'full_chart.png')

  # Check if the full chart exists
  if not os.path.exists(full_chart_path):
      print(f"Full chart not found for {folder_name}.")
      return

  # Load images for visualization
  original_image = Image.open(test_image_path)
  full_chart_image = Image.open(full_chart_path)

  # Set title color based on correct/incorrect prediction
  title_color = 'green' if random_class == predicted_class else 'red'

  # Plot the images side by side
  plt.figure(figsize=(10, 5))

  # Plot the test image
  plt.subplot(1, 2, 1)
  plt.imshow(original_image)
  plt.title("Test Image", fontsize=14)
  plt.axis('off')

  # Plot the full chart image
  plt.subplot(1, 2, 2)
  plt.imshow(full_chart_image)
  plt.title(f"Prediction: {predicted_class} ({confidence:.2f})", fontsize=14, color=title_color)
  plt.axis('off')

  # Display the plot
  plt.tight_layout()
  plt.show()


# In[16]:


# Add image paths to the dataset
def add_image_paths(example, split):
  """
  Add the correct file paths to the dataset for each example.
  """
  # Get the root directory for the split (train, validation, or test)
  root_dir = f"output_folder/{split}"

  # Extract the class folder name
  class_folder = dataset[split].features['label'].names[example['label']]

  # Construct the full file path using the image filename (already relative)
  example['image_path'] = f"{root_dir}/{class_folder}/{os.path.basename(example['image'].filename)}"
  return example


# In[17]:


def count_predictions_above_threshold(dataset, threshold):
  """
  Count how many predictions have a confidence score above a given threshold.

  Parameters:
      dataset (Dataset): The dataset to evaluate (e.g., test set).
      threshold (float): The minimum confidence score.

  Returns:
      int: The number of predictions above the threshold.
  """
  count = 0
  total = 0

  for item in dataset:
      # Get the image path
      image_path = item['image_path']

      # Perform prediction
      prediction = predict(image_path)
      confidence = prediction['confidence']

      # Increment count if confidence is above the threshold
      if confidence >= threshold:
          count += 1

      total += 1

  print(f"Out of {total} predictions, {count} are above the confidence threshold of {threshold:.2f}.")
  return count


# In[18]:


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
  return predicted_class, confidence


# In[19]:


# Function to visualize a random or specified historical_chart.png and full_chart.png
def visualize_sequence(base_dir, model, image_processor, id2label, csv_path, lot_size, folder_path=None):
  """
  Select a sequence folder (random or specified), predict on historical_chart.png,
  and display the historical chart and full chart with predictions and profit information.

  Args:
      base_dir (str): Base directory containing sequence folders.
      model (torch.nn.Module): Trained model for predictions.
      image_processor: Image processor for the model.
      id2label (dict): Mapping from class IDs to human-readable labels.
      csv_path (str): Path to the CSV file containing folder paths.
      folder_path (str, optional): Specific folder to visualize. If None, a random folder is selected.
  """
  # Read the CSV and extract the folder paths
  df = pd.read_csv(csv_path)

  # Ensure the 'Folder' column exists
  if "Folder" not in df.columns:
      raise ValueError("The 'Folder' column is missing in the CSV file.")

  # Get all sequence folders from the 'Folder' column
  sequence_folders = df['Folder'].tolist()

  # Determine the folder to use
  if folder_path:
      if folder_path not in sequence_folders:
          raise ValueError(f"Specified folder '{folder_path}' is not in the CSV file.")
      selected_folder = folder_path
  else:
      # Randomly select a sequence folder
      selected_folder = random.choice(sequence_folders)

  # Define file paths
  historical_chart_path = os.path.join(selected_folder, "historical_chart.png")
  full_chart_path = os.path.join(selected_folder, "full_chart.png")
  historical_data_path = os.path.join(selected_folder, "historical_data.csv")
  future_data_path = os.path.join(selected_folder, "future_data.csv")

  # Ensure necessary files exist
  if not all(
      os.path.exists(path)
      for path in [
          historical_chart_path,
          full_chart_path,
          historical_data_path,
          future_data_path,
      ]
  ):
      print(f"Missing necessary files in {selected_folder}.")
      return

  # Get the prediction for historical_chart.png
  predicted_class, confidence = predict_image(historical_chart_path, model, image_processor)
  predicted_label = id2label[predicted_class]

  # Read historical and future data
  historical_data = pd.read_csv(historical_data_path)
  future_data = pd.read_csv(future_data_path)

  # Get the last close price from historical and future data
  historical_last_close = historical_data['close'].iloc[-1]
  future_last_close = future_data['close'].iloc[-1]

  # Set stop loss and take profit thresholds
  stop_loss_limit = -100  # Maximum loss per trade
  take_profit_limit = 200  # Maximum profit per trade

  # Calculate profit or loss based on the predicted action
  if predicted_label == "buy":
    profit = (future_last_close - historical_last_close) * lot_size  # Profit for buy
  elif predicted_label == "sell":
    profit = (historical_last_close - future_last_close) * lot_size  # Profit for sell

  # Apply stop loss and take profit limits
  if profit <= stop_loss_limit:
      profit = stop_loss_limit
  elif profit >= take_profit_limit:
      profit = take_profit_limit

  # Load the images
  historical_chart = Image.open(historical_chart_path)
  full_chart = Image.open(full_chart_path)

  # Plot the images with prediction and profit
  plt.figure(figsize=(12, 6))

  # Historical chart
  plt.subplot(1, 2, 1)
  plt.imshow(historical_chart)
  plt.title(f"Prediction: {predicted_label} ({confidence:.2f})\nProfit: {profit:.2f}", fontsize=12)
  plt.axis("off")

  # Full chart
  plt.subplot(1, 2, 2)
  plt.imshow(full_chart)
  plt.title("Full Chart", fontsize=12)
  plt.axis("off")

  plt.tight_layout()
  plt.show()


# In[20]:


# Define a function to process and save predictions
def process_folders(base_dir, model, image_processor, output_csv_path):
  predictions_dict = {}
  total_folders = sum([len(dirs) for _, dirs, _ in os.walk(base_dir)])
  completed_folders = 0

  for root, dirs, _ in os.walk(base_dir):
      for dir_name in dirs:
          sequence_folder = os.path.join(root, dir_name)
          historical_chart_path = os.path.join(sequence_folder, "historical_chart.png")

          if os.path.exists(historical_chart_path):
              predicted_class, confidence = predict_image(historical_chart_path, model, image_processor)
              predictions_dict[sequence_folder] = {
                  "predicted_class": predicted_class,
                  "confidence": confidence,
              }

          completed_folders += 1
          progress_message = f"Progress: {completed_folders}/{total_folders} folders completed"
          sys.stdout.write(f"\r{progress_message}")
          sys.stdout.flush()

  print("\nProcessing complete!")

  with open(output_csv_path, mode="w", newline="") as csv_file:
      csv_writer = csv.writer(csv_file)
      csv_writer.writerow(["Folder", "Predicted Class", "Confidence"])
      for folder, result in predictions_dict.items():
          csv_writer.writerow([folder, result["predicted_class"], result["confidence"]])

  print(f"Results saved to {output_csv_path}")


# In[32]:


# Define a function to calculate profits and drawdowns
def calculate_profits_and_drawdowns(sorted_df, lot_size, starting_balance, output_folder):
  balance = starting_balance
  profits = []
  dates = []
  predicted_labels = []
  predictions_bool_list = []
  actual_labels = []
  folders = []

  id2label = {0: "buy", 1: "sell"}

  for folder in sorted_df['Folder']:
      historical_csv_path = os.path.join(folder, "historical_data.csv")
      future_csv_path = os.path.join(folder, "future_data.csv")

      if not os.path.exists(historical_csv_path) or not os.path.exists(future_csv_path):
          print(f"Missing data files in folder: {folder}")
          continue

      historical_data = pd.read_csv(historical_csv_path)
      future_data = pd.read_csv(future_csv_path)

      historical_last_close = historical_data['close'].iloc[-1]
      future_last_close = future_data['close'].iloc[-1]

      predicted_class = sorted_df.loc[sorted_df['Folder'] == folder, 'Predicted Class'].values[0]
      predicted_label = id2label[predicted_class]

      if predicted_label == "buy":
          profit = (future_last_close - historical_last_close) * lot_size
      elif predicted_label == "sell":
          profit = (historical_last_close - future_last_close) * lot_size

      balance += profit
      profits.append(balance)
      trade_date = future_data['time'].iloc[-1]
      dates.append(trade_date)
      predicted_labels.append(predicted_label)
      folders.append(folder)

      if (predicted_label == "buy" and future_last_close > historical_last_close) or \
          (predicted_label == "sell" and future_last_close < historical_last_close):
          predictions_bool_list.append(True)
      else:
          predictions_bool_list.append(False)

      actual_labels.append("buy" if future_last_close > historical_last_close else "sell")

  results_df = pd.DataFrame({
      "Date": dates,
      "Account Balance": profits,
      "Predicted Label": predicted_labels,
      "Actual Label": actual_labels,
      "Folder": folders,
      "Prediction Correct": predictions_bool_list,
  })

  max_drawdown = results_df['Account Balance'].cummax() - results_df['Account Balance']
  results_df['Drawdown'] = max_drawdown

  results_csv_path = os.path.join(output_folder, "trade_results.csv")
  results_df.to_csv(results_csv_path, index=False)
  #print(f"Results saved to {results_csv_path}")

  # Plot Account Balance
  plt.figure(figsize=(12, 6))
  plt.plot(pd.to_datetime(results_df["Date"]), results_df["Account Balance"], marker='o', linestyle='-')
  plt.title("Account Balance Over Time")
  plt.xlabel("Date")
  plt.ylabel("Account Balance")
  plt.axhline(starting_balance, color='blue', linestyle='--', label=f'Starting Balance: {starting_balance}')
  plt.axhline(balance, color='green', linestyle='-', label=f'Final Balance: {balance:.2f}')
  plt.legend()
  plt.grid()
  plt.tight_layout()
  account_balance_plot_path = os.path.join(output_folder, "account_balance_plot.png")
  plt.savefig(account_balance_plot_path)
  #plt.show()

  #print(f"Account balance plot saved to {account_balance_plot_path}")

  return results_df


# In[22]:


# Define a function to visualize random sequences
def visualize_sequence(folder, model, image_processor, id2label, lot_size, output_folder):
  historical_chart_path = os.path.join(folder, "historical_chart.png")
  full_chart_path = os.path.join(folder, "full_chart.png")
  historical_csv_path = os.path.join(folder, "historical_data.csv")
  future_csv_path = os.path.join(folder, "future_data.csv")

  if not all([os.path.exists(path) for path in [historical_chart_path, full_chart_path, historical_csv_path, future_csv_path]]):
      print(f"Missing files in {folder}")
      return

  predicted_class, confidence = predict_image(historical_chart_path, model, image_processor)
  predicted_label = id2label[predicted_class]

  historical_data = pd.read_csv(historical_csv_path)
  future_data = pd.read_csv(future_csv_path)

  historical_last_close = historical_data['close'].iloc[-1]
  future_last_close = future_data['close'].iloc[-1]

  if predicted_label == "buy":
      profit = (future_last_close - historical_last_close) * lot_size
  elif predicted_label == "sell":
      profit = (historical_last_close - future_last_close) * lot_size

  historical_chart = Image.open(historical_chart_path)
  full_chart = Image.open(full_chart_path)

  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.imshow(historical_chart)
  plt.title(f"Prediction: {predicted_label} ({confidence:.2f})\nProfit: {profit:.2f}")
  plt.axis("off")
  plt.subplot(1, 2, 2)
  plt.imshow(full_chart)
  plt.title("Full Chart")
  plt.axis("off")
  plt.tight_layout()

  visualization_path = os.path.join(output_folder, "random_sequence_visualization.png")
  plt.savefig(visualization_path)
  plt.show()

  print(f"Visualization saved to {visualization_path}")


# ## Load the model

# In[23]:


model_name = "facebook/convnext-base-224"


# In[24]:


# Define the configuration for the model
config = AutoConfig.from_pretrained(
    model_name,  # Use the base configuration from a pre-trained model
    num_labels=2,  # Number of classes
    id2label={0: "buy", 1: "sell"},
    label2id={"buy": 0, "sell": 1}
)

# Load the model with the defined configuration
model = AutoModelForImageClassification.from_config(config)

# Load the weights from the `safetensors` file
weights = load_file("/content/output_latest_model.zip/saved_model_1_no_threshold_based_model_64%/model.safetensors")
model.load_state_dict(weights)

print("Model weights loaded successfully.")

# Load dataset from folder structure
dataset = load_dataset('imagefolder', data_dir='/content/output_folder')

# Add image paths for each split in the dataset
for split in dataset.keys():
    dataset[split] = dataset[split].map(
        lambda example: add_image_paths(example, split),
        keep_in_memory=True
    )

# Verify the structure of the dataset
print(dataset['train'][0])  # Print the first example to check the 'image_path'
print(dataset['train'].features)

# Extract class names
class_names = dataset['train'].features['label'].names
print(f"Class names: {class_names}")

# Load a pretrained image processor
image_processor = AutoImageProcessor.from_pretrained(model_name)  # Updated to AutoImageProcessor

# Apply preprocessing to the dataset
dataset = dataset.with_transform(preprocess_images)

# Create DataLoader for the test set
test_loader = DataLoader(dataset['test'], batch_size=32)

print("Dataset and DataLoader initialized.")


# ## Evaluate the Model

# In[25]:


# Load class mappings
id2label = model.config.id2label
label2id = model.config.label2id

# Evaluate the model with a confidence threshold
threshold = 0.90
overall_report, filtered_report, y_pred_length_overall_results, y_pred_length_filtered_results = evaluate_model_with_threshold_report(model, test_loader, id2label, threshold)

# Print overall accuracy
print(f"Overall accuracy: {overall_report['accuracy'] * 100:.2f}%")

# Print filtered accuracy if available
if filtered_report:
    print(f"Filtered accuracy (Confidence > {threshold}): {filtered_report['accuracy'] * 100:.2f}%")
else:
    print("No predictions met the threshold.")

print(f"\nNumber of predictions above threshold {threshold}: {y_pred_length_filtered_results}/ {y_pred_length_overall_results}")


# In[26]:


test_image_base_path = '/content/output_folder/test'
candlestick_charts_path = '/content/output_latest_data.zip/candlestick_charts'
# Load class mappings
id2label = model.config.id2label
label2id = model.config.label2id
# Call the visualization function
visualize_prediction(threshold=0.0)


# In[27]:


# Call the visualization function
visualize_prediction(threshold)


# ## Downloading the Muliple Pairs data for predictions

# In[30]:


def download_and_extract_data_predictions(filename="output_latest", filename_output="output"):
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
          downloaded_file_path = kagglehub.dataset_download("sulimantadros/partly-multiple-pairs-1-hour")

          # Rename if necessary (if kagglehub returns a different name)
          if downloaded_file_path != filename:
              os.rename(downloaded_file_path, filename)
              print(f"Renamed downloaded file to '{filename}'")

      except Exception as e:
          print(f"An error occurred: {e}")
  else:
      print(f"File '{filename_output}' already exists. Skipping download and extraction.")

download_and_extract_data_predictions(filename="output_latest_data_part_multiple.zip", filename_output="output")


# ## Making Predictions on multiple pairs and evaluating the predictions

# In[33]:


# Define the base directory
base_directory = "/content/output_latest_data_part_multiple.zip/Pairs"

# Get the list of folder names
folder_names = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

output_folder_main = "output_backtesting_latest"
# Create the output directory if it doesn't exist
os.makedirs(output_folder_main, exist_ok=True)

# Define the model and image processor
config = AutoConfig.from_pretrained(model_name, num_labels=2, id2label={0: "buy", 1: "sell"}, label2id={"buy": 0, "sell": 1})
model = AutoModelForImageClassification.from_config(config)
weights = load_file("/content/output_latest_model.zip/saved_model_1_no_threshold_based_model_64%/model.safetensors")
model.load_state_dict(weights)
image_processor = AutoImageProcessor.from_pretrained(model_name)

starting_balance = 10000
lot_size = 10000

for folder_name in folder_names:
  print(f'Currently on Pair: {folder_name}')
  # Example usage
  base_dir = f"{base_directory}/{folder_name}"

  new_folder_path = os.path.join(output_folder_main, folder_name)
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
  results_df = calculate_profits_and_drawdowns(sorted_df, lot_size, starting_balance, new_folder_path)

  # Visualize a random sequence
  #random_folder = random.choice(sorted_df['Folder'].tolist())
  #visualize_sequence(random_folder, model, image_processor, {0: "buy", 1: "sell"}, lot_size, new_folder_path)

