from data.data_loader import get_data_loaders
from models.transformer.utils import get_device
from .general import load_trained_model
from config.utils import load_config
import matplotlib.pyplot as plt
import numpy as np
import torch


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
        i = 0
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
            actual_prices[:, :, i].flatten(), label=f"Actual {ohlc_labels[i]}", color='blue'
        )
        plt.plot(
            predicted_prices[:, :, i].flatten(), label=f"Predicted {ohlc_labels[i]}", color='orange', linestyle='dotted'
        )
        plt.title(f"{ohlc_labels[i]} : Actual vs Predicted")
        plt.xlabel("Time Steps")
        plt.ylabel(f"{ohlc_labels[i]}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def test_model_plot_window(configuration_file,weights_file,test_data_file):
    """
    @param configuration_file: config file name , under directory config
    @param weights_file: Weights file name , under directory weights/configuration_file
    @param test_data_file: processed test data file name , under directory data/processed
    """
    weights_path = f"../weights/{configuration_file}/{weights_file}"
    config_path = f"../config/{configuration_file}.json"
    # Load the model
    model = load_trained_model(weights_path, config_path)
    print("Model loaded successfully.")
    test_dataloader, _, _, _ = get_data_loaders(f"../data/processed/{test_data_file}",
                                          batch_size=64, config_file_path=config_path)
    infer_and_plot(model, test_dataloader, load_config(config_path)['seq_len_future'])
