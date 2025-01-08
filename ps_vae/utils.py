import yaml


def load_yaml_config(file_path: str) -> dict:
    """Loads config from yaml file
    Args:
        file_path (str): path to config file

    Returns:
        config (dict): config data
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    return config
