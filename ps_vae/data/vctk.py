import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Union, Callable
from ps_vae.utils import map_vctk_gender_to_label
import pandas as pd

METADATA_TRANSFORMS = {
    "gender": lambda x: map_vctk_gender_to_label(x['gender']),
}

class VCTKEmbeddingDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", metadata_transform: Union[str, None] = None):
        """
        Initialize the VCTK dataset.

        Args:
            data_root (str): Root directory containing speaker directories (e.g., "p001", "p002", ...).
            split (str): Data split indicator (unused in this version, kept for consistency).
            metadata_transform (str, optional): Metadata transformation to apply to each sample. Defaults to None.
        """
        self.data_root = data_root
        
         # Setup function to parse metadata
        if metadata_transform is not None:
            assert metadata_transform in METADATA_TRANSFORMS, f"Invalid metadata transform: {metadata_transform}"
            self.metadata_transform = METADATA_TRANSFORMS[metadata_transform]
        else:
            self.metadata_transform = None
            
        # Read metadata file and convert to a dictionary
        metadata_df = pd.read_csv(os.path.join(data_root, "vctk_metadata.csv"))
        self.metadata = {
            row['file_name']: {key: row[key] for key in row.index if key != 'file_name'}
            for _, row in metadata_df.iterrows()
        }
        
        # Build a list of tuples: (embedding_file_path, file_name)
        self.data_files = []
        for speaker_id in os.listdir(data_root):
            if speaker_id.startswith("p"):
                speaker_dir = os.path.join(data_root, speaker_id)
                for file_name in os.listdir(speaker_dir):
                    if file_name.endswith(".pt"):
                        self.data_files.append((os.path.join(speaker_dir, file_name), file_name))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        
        embed_file, file_name = self.data_files[idx]
        
        file_name = file_name.replace('_mic1.pt', '.wav')
        sample_data = self.metadata[file_name]

        if self.metadata_transform is not None:  # Apply transform if provided
            metadata = self.metadata_transform(sample_data)
            
        else:
            metadata = sample_data
            
        # Load embedding
        embed = torch.load(embed_file, weights_only=True)

        return embed.squeeze(), metadata

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
    dataset = VCTKEmbeddingDataset(**dataset_kwargs)
    
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

if __name__ == "__main__":
    
    import tqdm
    
    data_root = "/project/shrikann_35/nmehlman/psg_data/vctk_embeds/"
    
    dset = VCTKEmbeddingDataset(data_root, metadata_transform=None)
    print(len(dset))
    for x,y in tqdm.tqdm(dset):
        #print(x.shape, y)
        pass