import yaml
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from io import BytesIO
import PIL.Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

class LatentSpacePCACallback(pl.Callback):
    def __init__(self, dataloader, num_batches=None):
        """
        Callback to perform PCA on the latent space after each epoch.
        
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for the validation or test set.
            num_batches (int, optional): Number of batches to use for PCA computation. Defaults to using the whole dataset.
        """
        super().__init__()
        self.dataloader = dataloader
        self.num_batches = num_batches

    def on_validation_epoch_start(self, trainer, pl_module):
        """Perform PCA on the latent representations and plot after each epoch."""
        pl_module.eval()
        device = pl_module.device
        
        latent_vectors = []

        with torch.no_grad():
            for i, (batch, target) in enumerate(self.dataloader):
                if self.num_batches is not None and i >= self.num_batches:
                    break
                batch = batch.to(device)
                
                _, latent_repr, _ = pl_module(batch)  
                latent_vectors.append(latent_repr.cpu().numpy())

        # Convert to NumPy arrays
        latent_vectors = np.vstack(latent_vectors)

        # Perform PCA
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_vectors)

        # Plot results
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.7)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(f"Latent Space PCA - Epoch {trainer.current_epoch}")
        plt.grid(True)

        # Save plot
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = PIL.Image.open(buf)

        img_tensor = transforms.ToTensor()(img)

        # Log to TensorBoard
        trainer.logger.experiment.add_image("Latent Space PCA", img_tensor, global_step=trainer.current_epoch)

        plt.close()

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
    age_mapping = { 
        'teens': 0,
        'twenties': 0,
        'thirties': 1,
        'fourties': 1,
        'fifties': 1,
        'sixties': 2,
        'seventies': 2,
        'eighties': 2,
        'nineties': 2,
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

def map_vctk_gender_to_label(gender):
    """
    Maps a single gender category to a numerical label.

    Args:
        gender (str): Gender category as a string.

    Returns:
        int: Numerical gender label.
    """
    gender_mapping = {
        'M': 0,
        'F': 1,
    }
    return gender_mapping.get(gender, -1)  # Returns -1 if gender is not in the mapping

def sample_from_cv_metadata(metadata: pd.DataFrame, num_samples: int, conditions: dict = {}) -> pd.DataFrame:
    """
    Samples rows from the metadata DataFrame based on specified conditions.

    Args:
        metadata (pd.DataFrame): The metadata DataFrame to sample from.
        num_samples (int): The number of samples to retrieve.
        conditions (dict): A dictionary specifying conditions for filtering. 
                           Keys are column names, and values are lists of acceptable values or a single value.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled rows.

    Raises:
        ValueError: If there are not enough samples available for the given criteria.
    """
    filtered_metadata = metadata

    for key, value in conditions.items():
        if isinstance(value, list):
            filtered_metadata = filtered_metadata[filtered_metadata[key].isin(value)]
        else:
            filtered_metadata = filtered_metadata[filtered_metadata[key] == value]

    if len(filtered_metadata) < num_samples:
        raise ValueError(f"Not enough samples available for the given criteria. Available: {len(filtered_metadata)}, Required: {num_samples}")

    samples = filtered_metadata.sample(num_samples)

    return samples

if __name__ == "__main__":
    
    metadata = pd.read_csv("/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/train_embeds.tsv", sep="\t")
    conditions = {
        "age": ["twenties", "thirties"],
        "gender": "male"
    }

    num_samples = 10
    sampled_metadata = sample_from_cv_metadata(metadata, num_samples, conditions)
    print(sampled_metadata['path'])