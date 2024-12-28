import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from data.data_loader import get_data_loaders
from sklearn.metrics import mean_squared_error
from models.transformer.utils import get_device
from models.stock_predictor import StockPricePredictor



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

        for batch_input, batch_output, _, _ in dataloader:
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
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_epoch_{epoch+1}_mse_{best_loss}.pth"))
            print(f"Model saved at epoch {epoch+1}")

    return avg_epoch_loss

def evaluate_model(model, test_dataloader):
    model.eval()  # Set model to evaluation mode
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_input, batch_output, _, _ in test_dataloader:
            batch_size = len(batch_input)
            batch_input = batch_input.to(get_device())
            batch_output = batch_output.to(get_device())
            initial_ohlc = torch.randn(batch_size, 1, 4).to(get_device())
            # Generate predictions using the generate method
            predictions = model.generate(batch_input, future_steps=model.seq_len_future, initial_prices=initial_ohlc)

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
    #Get args
    parser = argparse.ArgumentParser(
        description= "Train the stock predictor model with the given configuration."
    )
    parser.add_argument("learning_rate", type=float, help="Learning rate for training.")
    parser.add_argument("num_epochs", type=int, help="Number of epochs for training.")
    parser.add_argument(
        "model_config_id",
        type=str,
        help="Name of the configuration file stored in the 'config' directory.",
    )

    args = parser.parse_args()

    # Initialize the model with hyperparameters
    config_id = args.model_config_id
    config_file_path = f"../config/{config_id}.json"

    config = load_config(config_file_path, verbose=True)
    weights_path = f"../weights/{config_id}/"
    path = Path(weights_path)
    path.mkdir(parents=True, exist_ok=True)
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
    train_dataloader, test_data_loader= get_data_loaders(
                processed_data_path= '../data/processed/China Merchants Bank_processed.csv',
                batch_size=64,
                config_file_path=config_file_path
        )

    train_loss = train_model(model, train_dataloader, num_epochs=args.num_epochs, lr=args.learning_rate, save_path=weights_path)

    # After training, you can evaluate the model
    mse = evaluate_model(model, test_data_loader)