import torch
from torch.utils.data import Dataset
import os
from typing import Dict, Union, Callable
from torch.utils.data import DataLoader, random_split
from ps_vae.utils import map_cv_gender_to_label, map_cv_age_to_label

METADATA_TRANSFORMS = {
    "gender": lambda x: map_cv_gender_to_label(x['gender']),
    "age": lambda x: map_cv_age_to_label(x['age']),
    "age_and_gender": lambda x: (map_cv_age_to_label(x['age']), map_cv_gender_to_label(x['gender']))
}

class CVEmbeddingDataset(Dataset):
    def __init__(
            self, data_root: str, split: str = "train", se_model: str = 'sb', metadata_transform: str = None, metadata_only: bool = False
    ):
        
        self.metadata_only = metadata_only
        
        # Setup function to parse metadata
        if metadata_transform is not None:
            assert metadata_transform in METADATA_TRANSFORMS, f"Invalid metadata transform: {metadata_transform}"
            self.metadata_transform = METADATA_TRANSFORMS[metadata_transform]
        else:
            self.metadata_transform = None

        # Read metadata file
        metadata_file = os.path.join(data_root, f"{split}.tsv")
        with open(metadata_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.metadata_headers = lines[0].strip().split("\t")
            self.metadata = [
                {
                    name: val
                    for name, val in zip(
                        self.metadata_headers, line.strip().split("\t")
                    )
                }
                for line in lines[1:]
            ]

        # Convert metadata to dict: {filename: info}
        self.metadata = {
            info.pop("path").replace(".mp3", ".pth"): info for info in self.metadata
        }

        # Will only load existing embeddings in directory, not all files in metadata

        self.embed_dir = os.path.join(data_root, f"embeds_{se_model}", split)
        self.embedding_files = [file for file in os.listdir(self.embed_dir) if file.endswith(".pth")]

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx: int):

        filename = self.embedding_files[idx]

        # Load metadata
        metadata = self.metadata[filename]

        if self.metadata_transform is not None:  # Apply transform if provided
            metadata = self.metadata_transform(metadata)

        if self.metadata_only:
            return metadata
        
        else:
            # Load embedding
            embed_file = os.path.join(self.embed_dir, filename)
            embed = torch.load(embed_file, weights_only=True)

            return embed.squeeze(), metadata


def get_cv_dataloaders(
    dataset_kwargs: Dict = {},
    batch_size: int = 16,
    collate_fn: Union[Callable, None] = None,
    train_frac: float = 1.0,
    **dataloader_kwargs,
) -> Union[DataLoader, Dict]:

    """Generate dataloader(s) for dataset_class with option to split into train/val

    Args:
        dataset_kwargs (Dict): kwargs for dataset construction
        batch_size (int): batch size
        collate_fn (Union[Callable, None], optional): Function to use for batch collation. Defaults to None.
        train_frac (float, optional): fraction of data to use for train split. Defaults to 1.0
        dataloader_kwargs (Dict, optional): additional kwargs to pass to dataloader constructor

    Returns:
        loader(s) (Union[ DataLoader, Dict ]): single dataloader if train_frac = 1.0, or dict with train/val loaders
        if train_frac < 1.0
    """

    dset = CVEmbeddingDataset(**dataset_kwargs)

    if train_frac < 1.0:

        dset_size = len(dset)
        train_size = int(dset_size * train_frac)
        val_size = dset_size - train_size

        train_dset, val_dset = random_split(dset, [train_size, val_size])

        train_loader = DataLoader(
            dataset=train_dset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        val_loader = DataLoader(
            dataset=val_dset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        loaders = {"train": train_loader, "val": val_loader}

        return loaders

    else:

        loader = DataLoader(
            dataset=dset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        return loader

