import unittest
import torch
from models.stock_predictor import StockPricePredictor

class TestStockPricePredictor(unittest.TestCase):
    def setUp(self):
        """Set up the model and parameters for testing."""
        self.feature_dim = 10    # Number of features in encoder input
        self.embed_dim = 16      # Embedding size
        self.seq_len_past = 20   # Length of historical input sequence
        self.seq_len_future = 5  # Length of future sequence
        self.num_heads = 2       # Number of attention heads
        self.num_layers = 2      # Number of transformer layers
        self.ff_dim = 64         # Feedforward network size
        self.dropout = 0.1       # Dropout rate

        # Instantiate the model
        self.model = StockPricePredictor(
            self.feature_dim, self.embed_dim, self.seq_len_past,
            self.seq_len_future, self.num_heads, self.num_layers,
            self.ff_dim, self.dropout
        )

    def test_forward_pass(self):
        """Test the forward pass of the model."""
        batch_size = 4
        # Generate dummy data
        encoder_input = torch.randn(batch_size, self.seq_len_past, self.feature_dim)  # Historical features
        decoder_input = torch.randn(batch_size, self.seq_len_future, 4)              # Future OHLC values

        # Forward pass
        output = self.model(encoder_input, decoder_input)

        # Assertions
        self.assertEqual(
            output.shape,
            (batch_size, self.seq_len_future, 4),
            f"Expected output shape (batch_size, seq_len_future, 4), got {output.shape}"
        )

    def test_generate_method(self):
        """Test the generate method of the model."""
        batch_size = 4
        seq_len_past = 30
        feature_dim = 6
        seq_len_future = 10

        # Create model instance
        model = StockPricePredictor(
            feature_dim=feature_dim,
            embed_dim=64,
            seq_len_past=seq_len_past,
            seq_len_future=seq_len_future,
            num_heads=4,
            num_layers=2,
            ff_dim=128
        )
        model.eval()  # Set to evaluation mode

        # Generate dummy data
        encoder_input = torch.randn(batch_size, seq_len_past, feature_dim)  # Historical features
        initial_ohlc = torch.randn(batch_size, 1, 4)  # Initial OHLC values
        future_steps = seq_len_future  # Number of future steps

        # Generate predictions
        with torch.no_grad():  # Ensure no gradients are calculated
            generated_output = model.generate(encoder_input, initial_ohlc, future_steps)

        # Assertions
        self.assertEqual(
            generated_output.shape,
            (batch_size, future_steps, 4),
            f"Expected generated output shape (batch_size, future_steps, 4), got {generated_output.shape}"
        )

        # Check values for anomalies
        self.assertFalse(
            torch.isnan(generated_output).any(),
            "Generated output contains NaN values."
        )
        self.assertFalse(
            torch.isinf(generated_output).any(),
            "Generated output contains infinite values."
        )


if __name__ == "__main__":
    unittest.main()
