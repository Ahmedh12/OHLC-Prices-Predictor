import os
import torch
import pandas as pd
from holoviews import output

from config.utils import load_config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class StockDataset(Dataset):
    def __init__(self, data, target_columns, history_length=30, forecast_length=5):
        """
        Custom Dataset for loading stock data for prediction with an autoregressive model.
        :param data: DataFrame with stock features
        :param target_columns: List of target column names (OHLC)
        :param history_length: Number of previous days
        :param forecast_length: Number of future days to predict
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
        # Ensure we have enough data for both input
        if idx + self.history_length + self.forecast_length <= len(self.data):
            input_slice = slice(idx, idx + self.history_length)
            output_slice = slice(idx + self.history_length, idx + self.history_length + self.forecast_length)
            x = self.data.iloc[input_slice][self.target_columns].values
            x = torch.tensor(x, dtype=torch.float32)


            y = self.data.iloc[output_slice][self.target_columns].values
            y = torch.tensor(y, dtype=torch.float32)

            x_dates = torch.tensor(self.data.iloc[input_slice]['Date'].astype('int64').values, dtype=torch.int64)
            y_dates = torch.tensor(self.data.iloc[output_slice]['Date'].astype('int64').values,dtype= torch.int64)

            return x, y , x_dates, y_dates
        else:
            # If not enough data, return None or skip this sequence
            return None

class StockValidationDataset(Dataset):
    def __init__(self, data, target_columns, history_length=30, forecast_length=5):
        """
        Custom Dataset for loading stock data for prediction with an autoregressive model.
        :param data: DataFrame with stock features
        :param target_columns: List of target column names (OHLC)
        :param history_length: Number of previous days
        :param forecast_length: Number of future days to predict
        """
        self.data = data
        self.target_columns = target_columns
        self.history_length = history_length
        self.forecast_length = forecast_length

    def __len__(self):
        # Length is adjusted to ensure non-overlapping sequences
        return (len(self.data) - self.history_length - self.forecast_length) // (self.history_length + self.forecast_length)

    def __getitem__(self, idx):
        # Adjust the index to make sure it jumps by `history_length + forecast_length` each time
        start_idx = idx * (self.history_length + self.forecast_length)

        # Ensure that there is enough data to slice for the given index
        if start_idx + self.history_length + self.forecast_length <= len(self.data):
            input_slice = slice(start_idx, start_idx + self.history_length)
            output_slice = slice(start_idx + self.history_length, start_idx + self.history_length + self.forecast_length)

            x = self.data.iloc[input_slice][self.target_columns].values
            x = torch.tensor(x, dtype=torch.float32)

            y = self.data.iloc[output_slice][self.target_columns].values
            y = torch.tensor(y, dtype=torch.float32)

            x_dates = torch.tensor(self.data.iloc[input_slice]['Date'].astype('int64').values, dtype=torch.int64)
            y_dates = torch.tensor(self.data.iloc[output_slice]['Date'].astype('int64').values, dtype=torch.int64)

            return x, y, x_dates, y_dates
        else:
            return None  # If not enough data, return None




def get_data_loaders(processed_data_path, config_file_path ,
                     batch_size=64,
                     train_test_split = 0.8,
                     target_columns : list = None):
    """
    Load the stock data from a CSV, split it into sequences (30 days of prices for input, 5 days for output),
    and return DataLoaders.
    :param processed_data_path: Path to processed stock data CSV
    :param config_file_path:  Path to the model parameters stored as json dict
    :param batch_size: Batch size for DataLoader
    :param train_test_split: ratio of training to test data
    :param target_columns: list of column name that are to be predicted
    :return: DataLoader objects for training and testing
    """
    # Load the processed data
    if target_columns is None:
        target_columns = ['Price', 'Open', 'High', 'Low']
    if os.path.exists(processed_data_path):
        data = pd.read_csv(processed_data_path)
        data["Date"] = pd.to_datetime(data["Date"])
    else:
        raise FileNotFoundError(f"File {processed_data_path} not found.")

    config = load_config(config_file_path)

    # Split data into train and test (you can also split based on time)
    train_size = int(train_test_split * len(data))  # 80% for training, 20% for testing
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Create datasets
    train_dataset = StockDataset(train_data, target_columns, history_length= config["seq_len_past"], forecast_length=config["seq_len_future"])
    test_dataset = StockDataset(test_data, target_columns, history_length= config["seq_len_past"], forecast_length=config["seq_len_future"])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_validation_dataloader(processed_data_path, config_file_path ,
                                 batch_size=64,
                                 target_columns : list = None):
    if target_columns is None:
        target_columns = ['Price', 'Open', 'High', 'Low']
    if os.path.exists(processed_data_path):
        data = pd.read_csv(processed_data_path)
        data["Date"] = pd.to_datetime(data["Date"])
    else:
        raise FileNotFoundError(f"File {processed_data_path} not found.")

    config = load_config(config_file_path)

    # Create datasets
    dataset = StockValidationDataset(data, target_columns, history_length=config["seq_len_past"], forecast_length=config["seq_len_future"])


    # Create DataLoaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader