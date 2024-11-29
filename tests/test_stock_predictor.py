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
        # Generate dummy data
        encoder_input = torch.randn(batch_size, self.seq_len_past, self.feature_dim)  # Historical features
        initial_ohlc = torch.randn(batch_size, 1, 4)                                  # Initial OHLC values
        future_steps = 10                                                             # Number of future steps

        # Generate predictions
        generated_output = self.model.generate(encoder_input, initial_ohlc, future_steps)

        # Assertions
        self.assertEqual(
            generated_output.shape,
            (batch_size, future_steps, 4),
            f"Expected generated output shape (batch_size, future_steps, 4), got {generated_output.shape}"
        )

if __name__ == "__main__":
    unittest.main()
