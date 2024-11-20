import json


def load_config(config_path):
    """
    Load a JSON configuration file.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        dict: Parsed configuration data as a Python dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: The file {config_path} was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
