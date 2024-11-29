from models.transformer.feedforward import FeedForward
import unittest
import torch


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.output_dim = 265
        self.dropout_rate = 0.05
        self.feedforward = FeedForward(self.embed_dim, self.output_dim, self.dropout_rate)

        self.batch_size = 16
        self.seq_len = 10

        self.dummy_input = torch.rand(self.batch_size, self.seq_len, self.embed_dim)

    def test_output_shape(self):
        result: torch.tensor = self.feedforward(self.dummy_input)
        self.assertTrue(result.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_no_nan_val(self):
        result = self.feedforward(self.dummy_input)
        self.assertFalse(torch.isnan(result).any(), " Output Contains NAN values")

    def test_deterministic_behavior(self):
        torch.manual_seed(42)
        res1 = self.feedforward(self.dummy_input)
        torch.manual_seed(42)
        res2 = self.feedforward(self.dummy_input)
        self.assertTrue(torch.equal(res1, res2), "FeedForward outputs differ for the same input and seed")


if __name__ == "__main__":
    unittest.main()