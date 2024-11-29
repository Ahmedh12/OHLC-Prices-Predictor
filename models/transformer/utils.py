import torch
def generate_causal_mask(size, batch_size = 1, num_heads = 1):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"