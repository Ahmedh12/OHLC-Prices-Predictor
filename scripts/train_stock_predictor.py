import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from sympy.core.random import shuffle

from models.transformer.utils import get_device
from models.stock_predictor import StockPricePredictor
from data.data_loader import get_data_loaders

import torch
import torch.optim as optim
import random

def train_model(model, dataloader, num_epochs=10, lr=0.001, teacher_forcing_ratio=0.5):
    # Set model to training mode
    model.train()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track metrics
    all_train_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_input, batch_output in dataloader:
            # Move data to the appropriate device (GPU/CPU)
            batch_input = batch_input.to(get_device())
            batch_output = batch_output.to(get_device())

            # Clear previous gradients
            optimizer.zero_grad()

            # Initialize the decoder input (start with zeros)
            decoder_input = torch.zeros(batch_input.size(0), model.seq_len_future, 4).to(get_device())

            # Determine whether to use teacher forcing
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Use ground truth as input to decoder during training (Teacher Forcing)
                outputs = model(batch_input, batch_output)
            else:
                # Use model's own previous prediction as input to decoder
                outputs = model(batch_input, decoder_input)

            # Calculate the loss
            loss = criterion(outputs, batch_output)
            epoch_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            num_batches += 1

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        all_train_loss.append(avg_epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    return all_train_loss


def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    all_true = []
    all_pred = []

    with torch.no_grad():  # No gradients needed during evaluation
        for batch_input, batch_output in dataloader:
            # Move data to the appropriate device (GPU/CPU)
            batch_input = batch_input.to(get_device())
            batch_output = batch_output.to(get_device())

            # Forward pass
            decoder_input = torch.zeros(batch_input.size(0), model.seq_len_future, 4).to(get_device())
            outputs = model(batch_input, decoder_input)

            # Collect predictions and true values
            all_true.append(batch_output.cpu())
            all_pred.append(outputs.cpu())

    # Stack the lists and calculate MSE
    all_true = torch.cat(all_true, dim=0).view(-1, 4)  # Flatten to (batch_size * future_steps, 4)
    all_pred = torch.cat(all_pred, dim=0).view(-1, 4)  # Flatten to (batch_size * future_steps, 4)

    # Calculate MSE using sklearn
    mse = mean_squared_error(all_true, all_pred)

    print(f"Test MSE: {mse:.4f}")
    return mse

# Example usage of the training and evaluation loop
if __name__ == "__main__":
    # Initialize the model with hyperparameters
    model = StockPricePredictor(
        feature_dim=4,  # e.g., 'Price', 'Open' 'High', 'Low', 'Vol.'
        embed_dim=64,
        seq_len_past=30,  # 30 days of history
        seq_len_future=5,  # 5 days forecast
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.1
    ).to(get_device())

    # Initialize DataLoader
    train_dataloader, test_data_loader = get_data_loaders('../data/processed/EGX 30 Historical Data_010308_280218_processed.csv', batch_size=64)
    train_loss = train_model(model, train_dataloader, num_epochs=25, lr=0.0005)

    # After training, you can evaluate the model
    mse = evaluate_model(model, test_data_loader)

    trdl, _ = get_data_loaders('../data/processed/EGX 30 Historical Data_010318_281124_processed.csv', batch_size=64)
    mse2 = evaluate_model(model, trdl)
