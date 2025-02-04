from ps_vae.data import CVEmbeddingDataset
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tqdm
from typing import Dict, Tuple, List


if __name__ == "__main__":
    # Define paths and dataset split
    data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    save_dir = "/home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/plots/"
    split = "train"
    N_SAMPLES = 1000

    # Load dataset
    dataset = CVEmbeddingDataset(data_root, split=split)

    # Process dataset
    norms = []
    idx = np.random.choice(len(dataset), N_SAMPLES, replace=False)
    for i in tqdm.tqdm(idx, desc='Processing samples'):
        sample = dataset[i][0].numpy()
        norms.append(np.linalg.norm(sample))
        
    plt.hist(norms, bins=50)
    plt.xlabel('Norm')
    plt.savefig('../plots/norms.png')
        
