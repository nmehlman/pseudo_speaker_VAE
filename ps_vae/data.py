import torch
from torch.utils.data import Dataset
import os

class CVEmbeddingDataset(Dataset):
    
    def __init__(self, data_root: str, split: str = 'train', metadata_transform: callable = None):
        
        self.metadata_transform = metadata_transform

        # Read metadata file
        metadata_file = os.path.join(data_root, f"{split}.tsv")
        with open(metadata_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.metadata_headers = lines[0].strip().split("\t")
            self.metadata = [ 
                    {
                    name: val for name, val in zip(self.metadata_headers, line.strip().split("\t"))
                } for line in lines[1:]
            ]

        # Convert metadata to dict: {filename: info}
        self.metadata = {info.pop('path').replace('.mp3', '.pth'): info for info in self.metadata}

        self.embed_dir = os.path.join(data_root, "embeds", split)
        self.embedding_files = os.listdir(self.embed_dir)

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):

        filename = self.embedding_files[idx]
        
        # Load embedding
        embed_file = os.path.join(self.embed_dir, filename)
        embed = torch.load(embed_file, weights_only=True)
        
        # Load metadata
        metadata = self.metadata[filename]

        if self.metadata_transform is not None: # Apply transform if provided
            metadata = self.metadata_transform(metadata)

        return embed.squeeze(), metadata
    
if __name__ == "__main__":
    data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    dataset = CVEmbeddingDataset(data_root, split="train")
    x, info = dataset[0]
    print(x.shape, info.keys())