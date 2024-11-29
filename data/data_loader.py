import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class StockDataset(Dataset):
    def __init__(self, data, target_columns, history_length=30, forecast_length=5):
        """
        Custom Dataset for loading stock data for prediction with an autoregressive model.
        :param data: DataFrame with stock features
        :param target_columns: List of target column names (OHLC)
        :param history_length: Number of previous days (30 days for input)
        :param forecast_length: Number of future days to predict (5 days for output)
        """
        self.data = data
        self.target_columns = target_columns
        self.history_length = history_length
        self.forecast_length = forecast_length

    def __len__(self):
        # Ensure that we have enough data for both the input and output sequences
        # If there are fewer than (history_length + forecast_length) rows left, we exclude that sequence
        return len(self.data) - self.history_length - self.forecast_length + 1

    def __getitem__(self, idx):
        """
        Get a sequence of 30 days of prices and the following 5 days of predictions.
        If there is not enough data to create a full sequence, this will return None.
        """
        # Ensure we have enough data for both input (30 days) and output (5 days)
        if idx + self.history_length + self.forecast_length <= len(self.data):
            # Input sequence: 30 days of previous data
            x = self.data.iloc[idx:idx + self.history_length][self.target_columns].values
            x = torch.tensor(x, dtype=torch.float32)

            # Target sequence: next 5 days of prices
            y = self.data.iloc[idx + self.history_length:idx + self.history_length + self.forecast_length][
                self.target_columns].values
            y = torch.tensor(y, dtype=torch.float32)

            return x, y
        else:
            # If not enough data, return None or skip this sequence
            return None


def get_data_loaders(processed_data_path, batch_size=64):
    """
    Load the stock data from a CSV, split it into sequences (30 days of prices for input, 5 days for output),
    and return DataLoaders.
    :param processed_data_path: Path to processed stock data CSV
    :param batch_size: Batch size for DataLoader
    :return: DataLoader objects for training and testing
    """
    # Load the processed data
    if os.path.exists(processed_data_path):
        data = pd.read_csv(processed_data_path)
    else:
        raise FileNotFoundError(f"File {processed_data_path} not found.")

    # Define target columns (OHLC)
    target_columns = ['Price', 'Open', 'High', 'Low']

    # Split data into train and test (you can also split based on time)
    train_size = int(0.8 * len(data))  # 80% for training, 20% for testing
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Create datasets
    train_dataset = StockDataset(train_data, target_columns)
    test_dataset = StockDataset(test_data, target_columns)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main():
    # Path to the raw stock data CSV file
    file_path = 'processed/EGX 30 Historical Data_010308_280218_processed.csv'  # Update this with the actual file path

    # Example columns (adjust as per your actual columns)
    target_columns = ['Price', 'Open', 'High', 'Low']  # These are OHLC columns

    # Load raw stock data into pandas DataFrame
    data = pd.read_csv(file_path)

    # Create the dataset using the StockDataset class
    history_length = 2  # 30 days of history
    forecast_length = 1  # Predict the next 5 days
    dataset = StockDataset(data, target_columns, history_length, forecast_length)

    # Create DataLoader for batching
    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size)

    # Test the data loader by iterating through the DataLoader
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if inputs is None or targets is None:
            continue  # Skip None batches

        print(f"Batch {batch_idx + 1}")
        print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        print(f"Inputs (first batch): {inputs[:5]}")
        print(f"Targets (first batch): {targets[:5]}")
        print("-" * 50)

        if batch_idx >= 3:  # Limiting to the first 3 batches for brevity
            break


if __name__ == "__main__":
    main()
