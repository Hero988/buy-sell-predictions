
# Buy-Sell Predictions

This repository contains code and scripts designed for predicting buy-sell signals in the Forex market using a machine learning algorithim. The implementation leverages machine learning models, specifically HuggingFace transformers, for image classification of candlestick chart data.

## Kaggle Links to Resources for my code
I set up different kaggle resources as I could not fit it into this github as the size of the data and the model is to much, please see below:
- Buy and Sell candlesticks: sulimantadros/buy-sell-object-detection-dataset (This dataset is all the data that is collected in the script test_4_object_detection_no_threshold.py, we then use this data to create a randomized training, testing and validation split)
- Trained Model: sulimantadros/hugging_face_model_facebookconvnext-base-224_64/other/default (This is the trained model that is was trained in the buy_sell_prediction-Using HuggingFace.ipynb code)
- Backtesting Data (01/01/2024 - 03/12/2024) - link to dataset in kaggle (This is the data that is collected in the test_4_object_detection_no_threshold.py code)
- Backtesting Results (01/01/2024 - 03/12/2024) - link to dataset in kaggle (These are the results from the backtest using the Backtesting Data which was generated from the backtesting_12_candles.py code)
- Backtesting Results Custom (01/01/2024 - 03/12/2024) - link to dataset in kaggle (This is the result where we find the best Backtesting Results of the top pairs and combine all the data and create the actual proper backtest which was generated from (enter code here))

## Results

You can find a folder called 'results' with the backtest for EURUSD for the past year from 2024-01-02 to 2024-11-29 

## Features

- **Backtesting Scripts**: Evaluate historical data to analyze model predictions.
- **Candlestick Chart Generation**: Automatically generate and save candlestick charts with indicators.
- **MetaTrader5 Integration**: Retrieve Forex data and execute trades programmatically.
- **Model Prediction**: Utilize pre-trained HuggingFace models for buy-sell signal classification.
- **Profit and Drawdown Analysis**: Calculate account performance based on model predictions.

## Requirements

### Python Libraries
- numpy
- pandas
- matplotlib
- mplfinance
- transformers
- safetensors
- MetaTrader5
- python-dotenv
- kagglehub
- torch
- scikit-learn
- PIL

### System Requirements
- MetaTrader5 installed with valid login credentials.
- Python 3.8 or higher.

## File Descriptions

### Jupyter Notebooks
- **buy_sell_prediction.ipynb**: Main notebook for training and testing the buy-sell prediction model.
- **Multiple_Pairs_buy_sell_prediction_backtesting.ipynb**: Script for backtesting multiple currency pairs using HuggingFace.

### Python Scripts
- **backtesting_12_candles.py**: Script for backtesting buy-sell predictions using historical candlestick chart data.
- **test_4_object_detection_no_threshold.py**: Generates and processes candlestick charts, with YOLO-compatible annotations for classification tasks.
- **test_4_predict_classification_backtesting.py**: Handles candlestick data for classification-based backtesting.
- **trading_forex.py**: Real-time prediction and trade execution using MetaTrader5.

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/buy-sell-predictions.git
    cd buy-sell-predictions
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure your MetaTrader5 environment variables in a `.env` file:
    ```plaintext
    MT5_LOGIN=your_login_id
    MT5_PASSWORD=your_password
    MT5_SERVER=your_server_name
    ```

4. Run the scripts or notebooks to train models, backtest, or execute live trades.

## Notes
- Ensure MetaTrader5 is properly initialized and logged in before running scripts.
- The pre-trained models are automatically downloaded from HuggingFace using the kagglehub library.

## Future Enhancements
- Expand model training capabilities with additional currency pairs.
- Add support for multi-timeframe analysis.
- Improve the efficiency of data handling and prediction pipelines.

## Contributing
Pull requests and issues are welcome. Please follow standard coding conventions and provide detailed descriptions for your contributions.

---
