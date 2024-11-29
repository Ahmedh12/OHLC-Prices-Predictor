import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import re


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


def preprocess_data(data):
    """
    Preprocess the raw stock data by handling missing values, feature scaling, etc.
    """
    # Convert Date to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    data = data.drop('Change %', axis=1)

    # Handle missing values, if any
    data = data.dropna(axis = 0, how = 'any')

    # Use MinMaxScaler to scale numerical features like 'Open', 'Close', etc.
    scaler = MinMaxScaler()
    scaled_columns = ['Price', 'Open', 'High', 'Low', 'Vol.']

    for col in scaled_columns:
        data.loc[:,col] = data[col].apply(convert_to_float)

    data.loc[:,scaled_columns] = scaler.fit_transform(data[scaled_columns])

    return data


def save_processed_data(data, save_path):
    """
    Save the processed data to a CSV file for later use.
    """
    data.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")



def main(raw_data_path, processed_data_path):
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
        processed_data = preprocess_data(raw_data)

        # Save the processed data
        save_processed_data(processed_data, processed_data_path)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    raw_data_file = 'raw/'
    processed_data_file = 'processed/'

    files = [
        "EGX 30 Historical Data_010308_280218",
        "EGX 30 Historical Data_010318_281124"
    ]

    # Call the main function with the paths
    for file in files:
        main(raw_data_file+file+".csv", processed_data_file+file+"_processed.csv")
