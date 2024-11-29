import unittest
import torch
from models.transformer.encoder import Encoder

class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.num_heads = 8
        self.hidden_dim = 256  # The hidden dimension for the feedforward network
        self.num_layers = 6    # Number of encoder layers
        self.batch_size = 16
        self.seq_len = 10
        self.dropout = 0.1

        # Initialize the Encoder
        self.encoder = Encoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        self.dummy_input = torch.rand(self.batch_size, self.seq_len, self.embed_dim)
        self.dummy_mask = torch.ones(self.batch_size, self.num_heads, self.seq_len, self.seq_len)

    def test_output_shape(self):
        output, _ = self.encoder(self.dummy_input, mask=self.dummy_mask)
        # Output shape should match (batch_size, seq_len, embed_dim)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_no_nan_in_output(self):
        output, _ = self.encoder(self.dummy_input, mask=self.dummy_mask)
        # Ensure there are no NaN values in the output
        self.assertFalse(torch.isnan(output).any())

    def test_deterministic_behavior(self):
        torch.manual_seed(42)
        output1, _ = self.encoder(self.dummy_input, mask=self.dummy_mask)
        torch.manual_seed(42)
        output2, _ = self.encoder(self.dummy_input, mask=self.dummy_mask)
        # Check if outputs are deterministic
        self.assertTrue(torch.equal(output1, output2))

    def test_effect_of_mask(self):
        # Mask out some positions (e.g., after index 5)
        self.dummy_mask[:, :, 5:] = 0
        output_with_mask, _ = self.encoder(self.dummy_input, mask=self.dummy_mask)
        output_without_mask, _ = self.encoder(self.dummy_input, mask=None)

        # Ensure masked positions influence the output
        self.assertFalse(torch.equal(output_with_mask, output_without_mask))

    def test_residual_connection(self):
        output, _ = self.encoder(self.dummy_input, mask=self.dummy_mask)
        # Check if the output retains elements of the input, indicative of residuals
        self.assertGreater(torch.norm(output - self.dummy_input), 0)

    def test_multiple_layers(self):
        # Test to ensure the number of layers affects the behavior
        encoder_single_layer = Encoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.hidden_dim,
            num_layers=1,  # Single layer encoder
            dropout=self.dropout
        )
        output_single_layer, _ = encoder_single_layer(self.dummy_input, mask=self.dummy_mask)
        output_multi_layer, _ = self.encoder(self.dummy_input, mask=self.dummy_mask)

        # Outputs from multi-layer and single-layer encoders should differ
        self.assertFalse(torch.equal(output_single_layer, output_multi_layer))


if __name__ == "__main__":
    unittest.main()
