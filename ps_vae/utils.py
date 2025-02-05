import yaml
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from io import BytesIO
import PIL.Image
import numpy as np

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

    def on_epoch_end(self, trainer, pl_module):
        """Perform PCA on the latent representations and plot after each epoch."""
        pl_module.eval()
        device = pl_module.device
        
        latent_vectors = []
        labels = []

        with torch.no_grad():
            for i, (batch, target) in enumerate(self.dataloader):
                if self.num_batches is not None and i >= self.num_batches:
                    break
                batch = batch.to(device)
                target = target.to(device)
                
                latent_repr = pl_module.get_latent_representation(batch)  # Assumes your model has this method
                latent_vectors.append(latent_repr.cpu().numpy())
                labels.append(target.cpu().numpy())

        # Convert to NumPy arrays
        latent_vectors = np.vstack(latent_vectors)
        labels = np.concatenate(labels)

        # Perform PCA
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_vectors)

        # Plot results
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=labels, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, label="Class Labels")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(f"Latent Space PCA - Epoch {trainer.current_epoch}")
        plt.grid(True)

        # Save plot
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = PIL.Image.open(buf)

        # Log to TensorBoard
        trainer.logger.experiment.add_image("Latent Space PCA", torch.tensor(np.array(img)), global_step=trainer.current_epoch)

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

