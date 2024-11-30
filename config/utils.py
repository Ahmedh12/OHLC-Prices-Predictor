import json

def load_config(config_path, verbose=False):
    """
    Load the model configuration from a JSON file.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    if verbose:
        print(f"Model parameters: {config}")

    return config
