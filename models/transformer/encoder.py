import torch.nn as nn
from ..transformer import MultiHeadAttention, FeedForward


class EncoderLayer(nn.Module):
    """
    A single encoder layer consisting of:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output, attention_weights = self.attention(x = x, self_mask = mask)
        x = self.layer_norm1(x+self.dropout(attention_output))

        ffn_output = self.ffn(x)
        x = self.layer_norm2(x+self.dropout(ffn_output))

        return x, attention_weights


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        attention_weights = []

        for layer in self.layers:
            x, weights = layer(x, mask)
            attention_weights.append(weights)

        return x, attention_weights
