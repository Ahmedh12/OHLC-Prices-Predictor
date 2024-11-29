import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout_rate = 0.1):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=ff_dim, out_features=embed_dim)
        )

    def forward(self,x):
        return self.feedforward(x)