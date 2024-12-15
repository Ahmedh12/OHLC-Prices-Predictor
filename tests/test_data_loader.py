import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.data_loader import StockDataset


class TestStockDataset(unittest.TestCase):
    def setUp(self):
        """
        Set up mock data for testing.
        """
        # Create a simple mock DataFrame for testing
        data = {
            'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'Open': [100 + i for i in range(100)],
            'High': [101 + i for i in range(100)],
            'Low': [99 + i for i in range(100)],
            'Close': [100 + i for i in range(100)],
            'Volume': [1000 + i * 10 for i in range(100)],
        }
        self.df = pd.DataFrame(data)

        # Target columns for OHLC data
        self.target_columns = ['Open', 'High', 'Low', 'Close']

        # History and forecast length
        self.history_length = 2  # Look back 30 days
        self.forecast_length = 1  # Forecast the next 5 days

    def test_dataset_length(self):
        """
        Test if the dataset length is correct.
        """
        # Create the dataset
        dataset = StockDataset(self.df, self.target_columns, self.history_length, self.forecast_length)

        # Length of the dataset should be len(df) - history_length
        expected_length = len(self.df) - self.history_length - self.forecast_length + 1
        self.assertEqual(len(dataset), expected_length)

    def test_dataloader_output(self):
        """
        Test the output of the DataLoader.
        """
        # Create the dataset
        dataset = StockDataset(self.df, self.target_columns, self.history_length, self.forecast_length)

        # Create DataLoader (no shuffling to respect the time series order)
        batch_size = 16
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Fetch one batch of data
        for batch_idx, (inputs, targets, _, _) in enumerate(dataloader):
            self.assertIsNotNone(inputs)
            self.assertIsNotNone(targets)
            self.assertEqual(inputs.shape[0], batch_size)  # Check if batch size is correct
            self.assertEqual(targets.shape[0], batch_size)
            self.assertEqual(inputs.shape[1], self.history_length)  # Length of the history
            self.assertEqual(inputs.shape[2], len(self.target_columns))  # Number of features (OHLC)
            self.assertEqual(targets.shape[1], self.forecast_length)  # Forecast length (5 days)
            break  # We only need to test the first batch for now

    def test_dataloader_no_shuffling(self):
        """
        Ensure that data is not shuffled, preserving the order of the time series.
        """
        # Create the dataset
        dataset = StockDataset(self.df, self.target_columns, self.history_length, self.forecast_length)

        # Create DataLoader (no shuffling)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Get the first two batches and check the order
        first_batch_input, _, _, dates = next(iter(dataloader))

        # Check if the second batch starts where the first batch ended (chronologically)
        self.assertTrue(torch.equal(first_batch_input[0][1:], first_batch_input[1][0:-1]))


if __name__ == '__main__':
    unittest.main()
