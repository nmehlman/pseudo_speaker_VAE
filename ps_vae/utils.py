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

def map_cv_age_to_label(age):
    """
    Maps a single age category to a numerical label.

    Args:
        age (str): Age category as a string.

    Returns:
        int: Numerical age label.
    """
    age_mapping = { #TODO update with all ages
        'teens': 0,
        'twenties': 1,
        'thirties': 2,
        'fourties': 3,
        'fifties': 4,
        'sixties': 5,
        'seventies': 6,
        'eighties': 7
    }
    return age_mapping.get(age, -1)  # Returns -1 if age is not in the mapping

def map_cv_gender_to_label(gender):
    """
    Maps a single gender category to a numerical label.

    Args:
        gender (str): Gender category as a string.

    Returns:
        int: Numerical gender label.
    """
    gender_mapping = {
        'male': 0,
        'female': 1,
        'other': 2
    }
    return gender_mapping.get(gender, -1)  # Returns -1 if gender is not in the mapping

