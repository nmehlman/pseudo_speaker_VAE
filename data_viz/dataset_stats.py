from ps_vae.data import CVEmbeddingDataset
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_sample(sample):
    sample_info = sample[1]
    age = sample_info["age"]
    gender = sample_info["gender"]
    accent = sample_info.get('accents', "unknown")
    return age, gender, accent


if __name__ == "__main__":
    # Load dataset
    data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    save_dir = "/home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/plots/"
    split = "train"
    dataset = CVEmbeddingDataset(data_root, split=split)

    # Set up ThreadPoolExecutor
    max_workers = min(32, os.cpu_count() + 4)  # Use a reasonable number of threads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in dataset}
        results = list(tqdm.tqdm(as_completed(futures), total=len(dataset), desc='Processing samples'))

    # Extract results
    processed_results = [future.result() for future in results]
    ages, genders, accents = zip(*processed_results)

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

    # Plot age distributions
    fig = plt.figure(figsize=(12, 4))

    # Age Distribution
    plt.subplot(1, 3, 1)
    counts = Counter(ages)
    categories = list(counts.keys())
    frequencies = list(counts.values())
    plt.bar(categories, frequencies)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title(f'Age Distribution in {split} split')

    # Gender Distribution
    plt.subplot(1, 3, 2)
    counts = Counter(genders)
    categories = list(counts.keys())
    frequencies = list(counts.values())
    plt.bar(categories, frequencies)
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    plt.title(f'Gender Distribution in {split} split')

    # Accent Distribution
    plt.subplot(1, 3, 3)
    counts = Counter(accents)
    categories = list(counts.keys())
    frequencies = list(counts.values())
    plt.bar(categories, frequencies)
    plt.xlabel('Accent')
    plt.ylabel('Frequency')
    plt.title(f'Accent Distribution in {split} split')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"traits_distribution_{split}_split.png"))
