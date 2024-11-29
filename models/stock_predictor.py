import torch
import torch.nn as nn
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.transformer.utils import generate_causal_mask, get_device

class StockPricePredictor(nn.Module):
    def __init__(self, feature_dim, embed_dim, seq_len_past, seq_len_future, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()

        self.seq_len_future = seq_len_future

        #Feature Extractor (we can try LSTMs here)
        self.encoder_embeddings = nn.Linear(feature_dim, embed_dim)
        self.decoder_embeddings = nn.Linear(4, embed_dim) #OHLC prices

        #Positional Encoding
        self.encoder_positional_encoding = nn.Parameter(torch.zeros(1, seq_len_past, embed_dim))
        self.decoder_positional_encoding = nn.Parameter(torch.zeros(1, seq_len_future, embed_dim))

        #Transformer
        self.encoder = Encoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        self.decoder = Decoder(num_layers, embed_dim, num_heads, ff_dim, dropout)

        #output Projection Layer
        self.output_layer = nn.Linear(embed_dim, 4)

    def forward(self, encoder_input, decoder_input):
        #encoder Processing
        encoder_embed = self.encoder_embeddings(encoder_input) + self.encoder_positional_encoding
        encoder_output, _ = self.encoder(encoder_embed)

        #decoder processing
        decoder_embed = self.decoder_embeddings(decoder_input) + self.decoder_positional_encoding
        decoder_output = self.decoder(decoder_embed, encoder_output, generate_causal_mask(self.seq_len_future).to(get_device()))

        #project Decoder output
        output = self.output_layer(decoder_output)

        return output

    def generate(self, encoder_input, initial_price, future_steps):
        """
        Generate future prices autoregressive.
        Args:
        - encoder_input: (batch_size, seq_len_past, feature_dim)
        - initial_price: (batch_size, 1, 1) Starting price for future predictions.
        - future_steps: int Number of future steps to predict.
        """
        # Encode past sequence
        encoder_emb = self.encoder_embeddings(encoder_input) + self.encoder_positional_encoding
        encoder_output, _ = self.encoder(encoder_emb)

        # Autoregressive decoding
        generated = []
        decoder_input = initial_price
        for _ in range(future_steps):
            decoder_emb = self.decoder_embeddings(decoder_input) + self.decoder_positional_encoding[:, :1, :]
            decoder_output = self.decoder(decoder_emb, encoder_output)
            next_price = self.output_layer(decoder_output[:, -1:, :])  # Predict next price
            generated.append(next_price)
            decoder_input = torch.cat((decoder_input, next_price), dim=1)  # Update decoder input

        return torch.cat(generated, dim=1)  # Concatenate all predictions




