from models.stock_predictor import StockPricePredictor
from models.transformer.utils import get_device
import matplotlib.pyplot as plt
from config.utils import load_config
import numpy as np
import torch

def load_trained_model(weights_path, config_file_path):

    config = load_config(config_file_path)

    model = StockPricePredictor(
        feature_dim=config["feature_dim"],
        embed_dim=config["embed_dim"],
        seq_len_past=config["seq_len_past"],
        seq_len_future=config["seq_len_future"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"]
    )

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))

    # Set model to evaluation mode
    model.eval()

    return model



def infer_and_plot(model, test_loader, seq_len_future):
    """
    Generates predictions from the model and plots them against actual data.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test set.
        seq_len_future: Number of future steps the model generates.
    """
    device = get_device()  # Ensure all tensors are on the same device
    model.to(device)  # Move model to the correct device
    model.eval()  # Set the model to evaluation mode

    actual_prices = []
    predicted_prices = []

    with torch.no_grad():
        for batch_input, batch_output in test_loader:

            batch_input = batch_input.to(device)
            batch_output = batch_output.to(device)

            # Get initial OHLC for generating future predictions
            initial_ohlc = batch_input[:, -1:, :]

            # Generate predictions
            predictions = model.generate(
                encoder_input=batch_input,
                initial_prices=initial_ohlc,
                future_steps=seq_len_future
            )

            # Collect real and predicted values
            actual_prices.append(batch_output.cpu().numpy())  # Move to CPU for plotting
            predicted_prices.append(predictions.cpu().numpy())  # Move to CPU for plotting

    # Convert lists to numpy arrays
    actual_prices = np.concatenate(actual_prices, axis=0)  # Combine the list into a single numpy array
    predicted_prices = np.concatenate(predicted_prices, axis=0)  # Combine the list into a single numpy array

    # Reshape for plotting (assuming 4 features: Open, High, Low, Close)
    actual_prices = actual_prices.reshape(-1, seq_len_future, 4)
    predicted_prices = predicted_prices.reshape(-1, seq_len_future, 4)

    # Plot OHLC predictions
    ohlc_labels = ['Price', 'Open', 'High', 'Low']
    for i in range(4):
        plt.figure(figsize=(10, 6))
        plt.plot(
            actual_prices[:, :, i].flatten(), label=f"Actual {ohlc_labels[i]}", color='blue', linestyle='--'
        )
        plt.plot(
            predicted_prices[:, :, i].flatten(), label=f"Predicted {ohlc_labels[i]}", color='orange'
        )
        plt.title(f"{ohlc_labels[i]} Prices: Actual vs Predicted")
        plt.xlabel("Time Steps")
        plt.ylabel(f"{ohlc_labels[i]} Price")
        plt.legend()
        plt.tight_layout()
        plt.show()
