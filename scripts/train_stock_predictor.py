import os
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
from models.transformer.utils import get_device
from models.stock_predictor import StockPricePredictor
from data.data_loader import get_data_loaders

import torch
import torch.optim as optim

from config.utils import load_config

def train_model(model, dataloader, num_epochs=10, lr=0.001, save_path="model.pth"):
    # Set model to training mode
    model.train()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')  # Initialize best_loss to a large value

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_input, batch_output in dataloader:
            batch_input = batch_input.to(get_device())
            batch_output = batch_output.to(get_device())

            optimizer.zero_grad()

            # Forward pass with teacher forcing
            decoder_input = torch.zeros(batch_input.size(0), model.seq_len_future, 4).to(get_device())  # Initialize decoder input (zero sequence)
            outputs = model(batch_input, decoder_input)

            # Calculate loss
            loss = criterion(outputs, batch_output)
            epoch_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        # Save the model if it achieves the best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_epoch{epoch+1}_mse{best_loss}.pth"))
            print(f"Model saved at epoch {epoch+1}")

    return avg_epoch_loss

def evaluate_model(model, test_dataloader):
    model.eval()  # Set model to evaluation mode
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_input, batch_output in test_dataloader:
            batch_size = len(batch_input)
            batch_input = batch_input.to(get_device())
            batch_output = batch_output.to(get_device())
            initial_ohlc = torch.randn(batch_size, 1, 4).to(get_device())
            # Generate predictions using the generate method
            predictions = model.generate(batch_input, future_steps = model.seq_len_future, initial_prices = initial_ohlc)

            all_true.append(batch_output.cpu().numpy())
            all_pred.append(predictions.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    all_true_flat = all_true.reshape(-1, all_true.shape[-1])
    all_pred_flat = all_pred.reshape(-1, all_pred.shape[-1])
    mse = mean_squared_error(all_true_flat, all_pred_flat)
    print(f"Test MSE: {mse:.4f}")

    return mse

# Example usage of the training and evaluation loop
if __name__ == "__main__":
    # Initialize the model with hyperparameters
    config = load_config("../config/config_1.json")

    model = StockPricePredictor(
        feature_dim=config["feature_dim"],
        embed_dim=config["embed_dim"],
        seq_len_past=config["seq_len_past"],
        seq_len_future=config["seq_len_future"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"]
    ).to(get_device())

    # Initialize DataLoader
    train_dataloader, test_data_loader = get_data_loaders('../data/processed/EGX 30 Historical Data_010308_280218_processed.csv', batch_size=64, config_file_path="../config/config_1.json")
    train_loss = train_model(model, train_dataloader, num_epochs=50, lr=0.00025, save_path="../weights/")

    # After training, you can evaluate the model
    mse = evaluate_model(model, test_data_loader)