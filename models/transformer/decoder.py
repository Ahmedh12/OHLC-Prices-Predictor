import torch
import torch.nn as nn
from mpmath import residual
from torch.nn.functional import dropout

from ..transformer import MultiHeadAttention, FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention                 = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.encoder_decoder_attention      = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn                            = FeedForward(embed_dim, ff_dim, dropout)
        self.layer_norm1                    = nn.LayerNorm(embed_dim)
        self.layer_norm2                    = nn.LayerNorm(embed_dim)
        self.layer_norm3                    = nn.LayerNorm(embed_dim)
        self.dropout1                       = nn.Dropout(dropout)
        self.dropout2                       = nn.Dropout(dropout)
        self.dropout3                       = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask = None, encoder_decoder_mask = None):
        #self-attention
        self_attention_outputs, _ = self.self_attention(x=x, self_mask= self_mask)
        x = self.layer_norm1(x+self.dropout1(self_attention_outputs))

        #encoder-decoder attention
        encoder_decoder_attention_outputs, _ = self.encoder_decoder_attention(x=x, encoder_output=encoder_output, cross_mask=encoder_decoder_mask)
        x = self.layer_norm2(x+self.dropout2(encoder_decoder_attention_outputs))

        # FeedForward
        x = self.layer_norm3(self.dropout3(self.ffn(x)))

        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_output, self_mask = None, encoder_decoder_mask = None):
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, encoder_decoder_mask)
        return self.layer_norm(x)






