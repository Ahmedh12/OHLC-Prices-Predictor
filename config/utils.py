import json
import os

def load_config(config_path, verbose=False):
    """
    Load the model configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File {config_path} not found.")

    with open(config_path, 'r') as file:
        config = json.load(file)
    if verbose:
        print(f"Model parameters: {config}")

    return config
