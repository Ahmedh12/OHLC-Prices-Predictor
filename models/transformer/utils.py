import torch
def generate_causal_mask(size, batch_size = 1, num_heads = 1):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

import torch

def get_relative_positional_encoding_tensor(max_seq_len, d_model):
    """
    Generate relative positional encodings to add to input embeddings.

    Args:
    - max_seq_len (int): Maximum sequence length.
    - d_model (int): Dimensionality of the model embeddings.

    Returns:
    - torch.Tensor: Relative positional encodings of shape (max_seq_len, d_model).
    """
    # Create a relative position matrix: (max_seq_len, max_seq_len)
    positions = torch.arange(-max_seq_len + 1, max_seq_len).unsqueeze(1)  # Relative positions
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

    # Sinusoidal encoding based on relative positions
    sin_enc = torch.sin(positions * div_term)  # (2 * max_seq_len - 1, d_model / 2)
    cos_enc = torch.cos(positions * div_term)  # (2 * max_seq_len - 1, d_model / 2)

    # Concatenate sine and cosine encodings
    rel_positional_encodings = torch.cat([sin_enc, cos_enc], dim=-1)  # (2 * max_seq_len - 1, d_model)

    # Extract encodings for positions within the sequence
    rel_positional_encodings = rel_positional_encodings[max_seq_len - 1 : max_seq_len - 1 + max_seq_len]

    return rel_positional_encodings.expand(1,-1,-1)
