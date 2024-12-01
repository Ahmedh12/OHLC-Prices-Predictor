import os
import re
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def convert_to_float(value):
    """
    Converts a string value like '31.88M' to float (in this case, multiplying by 1 million).
    """
    if isinstance(value, str):
        value = value.replace(',', '')  # Remove commas if any
        # Regex to capture the number and suffix
        match = re.match(r"([0-9.]+)([a-zA-Z]+)?", value)
        if match:
            number, suffix = match.groups()
            number = float(number)

            if suffix == 'M':
                return number * 1e6
            elif suffix == 'K':
                return number * 1e3
            elif suffix == 'B':
                return number * 1e9
        else:
            return float(value)
    return float(value)

def load_data(file_path):
    """
    Load the stock price data from a CSV file.
    """
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data
    else:
        raise FileNotFoundError(f"File {file_path} not found.")

def preprocess_data(data, scaler_save_path=None):
    """
    Preprocess the raw stock data by handling missing values, feature scaling, etc.
    """
    # Convert Date to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    data.drop('Change %', axis=1 , inplace=True)

    # Handle missing values, if any
    data.dropna(axis = 0, how = 'any', inplace=True)

    scaler = RobustScaler()
    scaled_columns = ['Price', 'Open', 'High', 'Low', 'Vol.']

    for col in scaled_columns:
        data.loc[:,col] = data[col].apply(convert_to_float)

    data.loc[:,scaled_columns] = scaler.fit_transform(data[scaled_columns])

    data.sort_values(by='Date', ascending=True, inplace=True)

    if scaler_save_path is not None:
        with open(scaler_save_path, "wb") as f:
            pickle.dump(scaler,f)

    return data

def save_processed_data(data, save_path):
    """
    Save the processed data to a CSV file for later use.
    """
    data.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")



def main(raw_data_path, processed_data_path, scaler_save_path = None):
    """
    Main function that loads raw data, preprocesses it, and saves the processed data.

    Args:
        raw_data_path (str): Path to the raw stock price data CSV.
        processed_data_path (str): Path where the processed data should be saved.
    """
    try:
        # Load the raw data
        raw_data = load_data(raw_data_path)

        # Preprocess the data
        processed_data = preprocess_data(raw_data, scaler_save_path)

        # Save the processed data
        save_processed_data(processed_data, processed_data_path)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    raw_data_folder = 'raw'
    processed_data_folder = 'processed'
    scalers_folder = 'scalers'

    files = [
        "EGX 30 Historical Data_010308_280218",
        "EGX 30 Historical Data_010318_281124",
        "BTC_USD Bitfinex Historical Data_010308_280218",
        "BTC_USD Bitfinex Historical Data_010318_281124",
    ]

    # Call the main function with the paths
    for file in files:
        main(f"{raw_data_folder}/{file}.csv", f"{processed_data_folder}/{file}_processed.csv", f"{scalers_folder}/{file}_scaler.pkl")
