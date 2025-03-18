import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Union, Callable

class VCTKDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train"):
        """
        Initialize the VCTK dataset.

        Args:
            data_root (str): Root directory containing speaker directories (e.g., "p001", "p002", ...).
            split (str): Data split indicator (unused in this version, kept for consistency).
        """
        self.data_root = data_root
        # Build a list of tuples: (embedding_file_path, speaker_id)
        self.data_files = []

        # Loop over each speaker directory in the data_root that follows the "pXXX" pattern.
        for speaker_dir in sorted(os.listdir(data_root)):
            speaker_path = os.path.join(data_root, speaker_dir)
            if os.path.isdir(speaker_path) and speaker_dir.startswith("p") and speaker_dir[1:].isdigit():
                # Extract speaker ID by stripping the 'p' and converting the rest to an integer.
                speaker_id = int(speaker_dir[1:])
                # List all .pkl files in the speaker directory.
                embedding_files = [f for f in os.listdir(speaker_path) if f.endswith(".pt")]
                embedding_files.sort()
                # Append each file's full path along with the speaker ID.
                for file in embedding_files:
                    full_path = os.path.join(speaker_path, file)
                    self.data_files.append((full_path, speaker_id))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx: int):
        embedding_file, speaker_id = self.data_files[idx]
        # Load the embedding from the pkl file.
        with open(embedding_file, "rb") as f:
            embedding = torch.load(f, map_location='cpu', weights_only=False).squeeze()
        return embedding, speaker_id

def get_vctk_dataloaders(
    dataset_kwargs: Dict = {},
    batch_size: int = 16,
    collate_fn: Union[Callable, None] = None,
    train_frac: float = 1.0,
    **dataloader_kwargs,
) -> Union[DataLoader, Dict]:
    """
    Generate dataloader(s) for the VCTKDataset with an option to split into train/validation.

    Args:
        dataset_kwargs (Dict): kwargs for dataset construction.
        batch_size (int): Batch size.
        collate_fn (Callable, optional): Function to use for batch collation. Defaults to None.
        train_frac (float, optional): Fraction of data to use for training (if less than 1.0, validation is created).
        dataloader_kwargs: Additional kwargs to pass to the DataLoader constructor.

    Returns:
        Union[DataLoader, Dict]: A single DataLoader if train_frac == 1.0, or a dictionary with "train" and "val" loaders.
    """
    dataset = VCTKDataset(**dataset_kwargs)
    
    if train_frac < 1.0:
        dataset_size = len(dataset)
        train_size = int(dataset_size * train_frac)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn, **dataloader_kwargs
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collate_fn, **dataloader_kwargs
        )

        return {"train": train_loader, "val": val_loader}
    else:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, **dataloader_kwargs
        )
