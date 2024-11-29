import unittest
import torch
from models.transformer.decoder import Decoder  # Assuming the Decoder is implemented in decoder.py

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.num_heads = 8
        self.ff_dim = 256
        self.num_layers = 6
        self.batch_size = 16
        self.seq_len = 10
        self.encoder_seq_len = 15
        self.dropout = 0.1

        self.decoder = Decoder(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        )
        self.dummy_decoder_input = torch.rand(self.batch_size, self.seq_len, self.embed_dim)
        self.dummy_encoder_output = torch.rand(self.batch_size, self.encoder_seq_len, self.embed_dim)
        self.self_mask = torch.ones(self.seq_len, self.seq_len)
        self.cross_mask = torch.ones(self.seq_len, self.encoder_seq_len)

    def test_output_shape(self):
        output = self.decoder(self.dummy_decoder_input, self.dummy_encoder_output, self.self_mask, self.cross_mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_masked_attention(self):
        # Apply a self_mask that ignores positions 3 onwards
        self.self_mask[3:, :] = 0
        output = self.decoder(self.dummy_decoder_input, self.dummy_encoder_output, self.self_mask, self.cross_mask)
        self.assertFalse(torch.isnan(output).any())  # Ensure no NaNs in the output

    def test_deterministic_behavior(self):
        torch.manual_seed(42)
        output1 = self.decoder(self.dummy_decoder_input, self.dummy_encoder_output, self.self_mask, self.cross_mask)
        torch.manual_seed(42)
        output2 = self.decoder(self.dummy_decoder_input, self.dummy_encoder_output, self.self_mask, self.cross_mask)
        self.assertTrue(torch.equal(output1, output2))

    def test_no_nan_in_output(self):
        output = self.decoder(self.dummy_decoder_input, self.dummy_encoder_output, self.self_mask, self.cross_mask)
        self.assertFalse(torch.isnan(output).any())

if __name__ == "__main__":
    unittest.main()
