from ps_vae.data import CVEmbeddingDataset
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


if __name__ == "__main__":
   
    # Load dataset
    data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    save_dir = "/Users/nmehlman/Desktop/pseudo_speaker_VAE/plots/"
    split = "train"
    dataset = CVEmbeddingDataset(data_root, split=split)
    
    ages = []
    genders = []
    accents = []
    for _, sample_info in dataset:
        
        ages.append(sample_info["age"])
        genders.append(sample_info["gender"])
        accents.append(sample_info.get('accent', "unknown"))
        
    # Get unique values and save to json
    unique_ages = list(set(ages))
    unique_genders = list(set(genders))
    unique_accents = list(set(accents))
    
    info = {
        "unique_ages": unique_ages,
        "unique_genders": unique_genders,
        "unique_accents": unique_accents
    }
    
    with open(os.path.join(save_dir, f"dataset_info_{split}.json"), "w") as f:
        json.dump(info, f, indent=4)
        
    ## Plot age distributions
    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    counts = Counter(ages)

    categories = list(counts.keys())
    frequencies = list(counts.values())

    plt.bar(categories, frequencies)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title(f'Age Distribution in {split} split') 
    ##
    
    ## Plot gender distribution
    plt.subplot(1, 3, 2)
    counts = Counter(genders)

    categories = list(counts.keys())
    frequencies = list(counts.values())

    plt.bar(categories, frequencies)
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    plt.title(f'Gender Distribution in {split} split') 
    ##
    
    ## Plot accent distribution
    plt.subplot(1, 3, 3)
    counts = Counter(accents)

    categories = list(counts.keys())
    frequencies = list(counts.values())

    plt.bar(categories, frequencies)
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    plt.title(f'Accent Distribution in {split} split') 
    
    plt.savefig(os.path.join(save_dir, f"traits_distribution_{split}_split.png"))