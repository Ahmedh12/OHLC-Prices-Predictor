import torch
import torch.nn as nn
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.transformer.utils import generate_causal_mask, get_device, get_relative_positional_encoding_tensor

class StockPricePredictor(nn.Module):
    def __init__(self, feature_dim, embed_dim, seq_len_past, seq_len_future, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()

        self.seq_len_future = seq_len_future

        #Feature Extractor (we can try LSTMs here)
        self.encoder_embeddings = nn.Linear(feature_dim, embed_dim)
        self.decoder_embeddings = nn.Linear(4, embed_dim) #OHLC prices

        #Positional Encoding
        # self.encoder_positional_encoding = nn.Parameter(get_relative_positional_encoding_tensor(seq_len_past, embed_dim))
        # self.decoder_positional_encoding = nn.Parameter(get_relative_positional_encoding_tensor(seq_len_future, embed_dim))

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

    def generate(self, encoder_input, initial_prices, future_steps):
        """
        Generate future OHLC prices autoregressively.

        Args:
        - encoder_input: (batch_size, seq_len_past, feature_dim) Past price history.
        - initial_prices: (batch_size, 1, 4) Initial OHLC prices to start predictions.
        - future_steps: int, Number of future steps to predict.

        Returns:
        - Tensor of shape (batch_size, future_steps, 4) with predicted OHLC values.
        """
        # Encode past sequence
        encoder_emb = self.encoder_embeddings(encoder_input) + self.encoder_positional_encoding
        encoder_output, _ = self.encoder(encoder_emb)

        # Initialize variables for autoregressive decoding
        predictions = []
        decoder_input = initial_prices  # Start with the provided initial OHLC prices

        for step in range(future_steps):
            # Embed the decoder input
            decoder_emb = self.decoder_embeddings(decoder_input) + self.decoder_positional_encoding[:,
                                                                   :decoder_input.size(1), :]

            # Decode using the encoder's output and the decoder's previous outputs
            decoder_output = self.decoder(decoder_emb, encoder_output)

            # Predict the next OHLC values
            next_ohlc = self.output_layer(decoder_output[:, -1:, :])  # Only the last time step

            # Append the prediction
            predictions.append(next_ohlc)

            # Update decoder input with the new OHLC value
            decoder_input = torch.cat((decoder_input, next_ohlc), dim=1)

        # Concatenate all predictions along the time dimension
        return torch.cat(predictions, dim=1)



