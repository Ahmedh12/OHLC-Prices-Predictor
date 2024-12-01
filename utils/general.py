import torch
import pandas as pd
from config.utils import load_config
from models.stock_predictor import StockPricePredictor

def convertINT64ToDateStr(int64_date : list[int]):
    return pd.to_datetime(int64_date)

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


