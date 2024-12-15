from .general import load_trained_model, convertINT64ToDateTimeObj
from data.data_loader import  get_validation_dataloader
from models.transformer.utils import get_device
from.evaluationMetrics import getEvalMetrics, printEvalMetrics
from config.utils import load_config
import matplotlib.pyplot as plt
import numpy as np
import torch

def infer(model,
           test_loader,
           seq_len_future,
           num_features=4,
           pred_cycle_count=0,
           predict_earliest=True):

    device = get_device()  # Ensure all tensors are on the same device
    model.to(device)  # Move model to the correct device
    model.eval()  # Set the model to evaluation mode

    actual_prices = []
    predicted_prices = []
    dates = []
    if pred_cycle_count <= 0:
        pred_cycle_count = len(test_loader)
    elif pred_cycle_count > len(test_loader):
        pred_cycle_count = len(test_loader)

    def skip_windows():
        if predict_earliest:
            return idx >= pred_cycle_count
        else:
            return idx < len(test_loader) - pred_cycle_count

    with torch.no_grad():
        for idx, (batch_input, batch_output, _, output_dates) in enumerate(test_loader):
            if predict_earliest and skip_windows():  # predict from the end of the year
                break
            elif not predict_earliest and skip_windows():
                continue

            batch_input = batch_input.to(device)
            batch_output = batch_output.to(device)
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
            dates.append(output_dates.cpu().numpy())

    # Convert lists to numpy arrays
    actual_prices = np.concatenate(actual_prices, axis=0)
    predicted_prices = np.concatenate(predicted_prices, axis=0)
    dates = np.concatenate(dates, axis=0)

    actual_prices = actual_prices.reshape(-1, seq_len_future, num_features)
    predicted_prices = predicted_prices.reshape(-1, seq_len_future, num_features)
    dates = convertINT64ToDateTimeObj(dates.reshape(-1))

    return dates, actual_prices, predicted_prices



def plot(plot_labels,
         num_features,
         dates,
         actual_prices,
         predicted_prices):

    for i in range(num_features):
        plt.figure(figsize=(10, 6))
        plt.plot(
            dates, actual_prices[:, :, i].flatten(), label=f"Actual {plot_labels[i]}", color='blue', linewidth=1
        )
        plt.plot(
            dates, predicted_prices[:, :, i].flatten(), label=f"Predicted {plot_labels[i]}", linewidth=3,
            color='orange', linestyle='dotted'
        )
        plt.title(f"{plot_labels[i]} : Actual vs Predicted")
        plt.xlabel("Time Steps")
        plt.ylabel(f"{plot_labels[i]}")
        plt.legend()
        plt.show()

def infer_and_plot(model,
                   test_loader,
                   seq_len_future,
                   num_features=4,
                   plot_labels=None,
                   pred_cycle_count=0,
                   predict_earliest=True):

    if plot_labels is None:
        plot_labels = ['Price', 'Open', 'High', 'Low']

    dates, actual_prices, predicted_prices = infer(model            = model,
                                                   test_loader      = test_loader,
                                                   seq_len_future   = seq_len_future,
                                                   num_features     = num_features,
                                                   pred_cycle_count = pred_cycle_count,
                                                   predict_earliest = predict_earliest)

    plot(
        plot_labels     = plot_labels,
        num_features    = num_features,
        dates           = dates,
        actual_prices   = actual_prices,
        predicted_prices= predicted_prices)

    return [(plot_labels[i], actual_prices[:, :, i].flatten(), predicted_prices[:, :, i].flatten()) for i in range(num_features)]

def test_model_plot_window(configuration_file,
                           weights_file,
                           test_data_file,
                           pred_cycle_count=0,
                           predict_earliest=True,
                           print_eval_metrics = True):
    """
    @param configuration_file: config file name , under directory config
    @param weights_file: Weights file name , under directory weights/configuration_file
    @param test_data_file: processed test data file name , under directory data/processed
    """
    weights_path = f"../weights/{configuration_file}/{weights_file}"
    config_path = f"../config/{configuration_file}.json"
    # Load the model
    model   = load_trained_model(weights_path, config_path)
    config  = load_config(config_path)
    print("Model loaded successfully.")
    loader = get_validation_dataloader(f"../data/processed/{test_data_file}",
                                          batch_size=1, config_file_path=config_path)
    pred_list = infer_and_plot(model,
                               loader,
                               config['seq_len_future'],
                               pred_cycle_count = pred_cycle_count,
                               predict_earliest = predict_earliest)

    if print_eval_metrics:
        for i in range(config['feature_dim']):
            printEvalMetrics(*getEvalMetrics(pred_list[i]))