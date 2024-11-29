import unittest
import torch
from models.transformer.attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 512
        self.num_heads = 8
        self.batch_size = 64
        self.seq_len = 10
        self.dropout = 0.1

        self.attention = MultiHeadAttention(self.embed_dim, self.num_heads, self.dropout)
        self.dummy_input = torch.rand(self.batch_size, self.seq_len, self.embed_dim)

    def test_output_shape(self):
        output, weights = self.attention(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))

    def test_no_nan_in_output(self):
        output, _ = self.attention(self.dummy_input)
        self.assertFalse(torch.isnan(output).any())

    def test_deterministic_behavior(self):
        torch.manual_seed(42)
        output1, _ = self.attention(self.dummy_input)
        torch.manual_seed(42)
        output2, _ = self.attention(self.dummy_input)
        self.assertTrue(torch.equal(output1, output2))

    def test_masked_attention(self):
        # Create a self_mask with 1s for positions to keep and 0s for positions to ignore
        mask = torch.ones(self.seq_len, self.seq_len)  # (batch_size, num_heads, seq_len, seq_len)
        mask[1:, :] = 0
        # Forward pass with the self_mask
        _, attention_weights = self.attention(self.dummy_input, self_mask=mask)

        attention_scores = attention_weights[:, :, 1:, :]
        self.assertTrue(torch.equal(attention_scores, torch.zeros_like(attention_scores)))



if __name__ == "__main__":
    unittest.main()