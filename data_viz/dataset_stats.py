from ps_vae.data import CVEmbeddingDataset
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tqdm


def process_sample(sample_info):
    age = sample_info["age"]
    gender = sample_info["gender"]
    accent = sample_info["accents"]
    if accent == "": accent = "Unknown"
    return age, gender, accent


if __name__ == "__main__":
    # Load dataset
    data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    save_dir = "/home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/plots/"
    split = "train"
    dataset = CVEmbeddingDataset(data_root, split=split, metadata_only=True)

    # Process dataset with a simple for loop
    results = []
    for sample in tqdm.tqdm(dataset, total=len(dataset), desc='Processing samples'):
        results.append(process_sample(sample))

    # Extract results
    ages, genders, accents = zip(*results)

    # Get unique values and save to json
    unique_ages = list(set(ages))
    unique_genders = list(set(genders))
    unique_accents = list(set(accents))

    # Calculate counts and percentages
    age_counts = Counter(ages)
    gender_counts = Counter(genders)
    accent_counts = Counter(accents)

    total_samples = len(ages)
    age_percentages = {age: count / total_samples * 100 for age, count in age_counts.items()}
    gender_percentages = {gender: count / total_samples * 100 for gender, count in gender_counts.items()}
    accent_percentages = {accent: count / total_samples * 100 for accent, count in accent_counts.items()}

    info = {
        "unique_ages": unique_ages,
        "unique_genders": unique_genders,
        "unique_accents": unique_accents,
        "age_counts": age_counts,
        "gender_counts": gender_counts,
        "accent_counts": accent_counts,
        "age_percentages": age_percentages,
        "gender_percentages": gender_percentages,
        "accent_percentages": accent_percentages
    }

    with open(os.path.join(save_dir, f"dataset_info_{split}.json"), "w") as f:
        json.dump(info, f, indent=4)

    # Plot age distributions
    fig = plt.figure(figsize=(15, 4))

    # Age Distribution
    plt.subplot(1, 3, 1)
    counts = Counter(ages)
    categories = list(counts.keys())
    frequencies = list(counts.values())
    plt.bar(categories, frequencies)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title(f'Age Distribution in {split} split')
    plt.xticks(rotation=45)

    # Gender Distribution
    plt.subplot(1, 3, 2)
    counts = Counter(genders)
    categories = list(counts.keys())
    frequencies = list(counts.values())
    plt.bar(categories, frequencies)
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    plt.title(f'Gender Distribution in {split} split')
    plt.xticks(rotation=45)

    # Accent Distribution (only top 10)
    plt.subplot(1, 3, 3)
    counts = Counter(accents)
    most_common_accents = counts.most_common(10)
    categories, frequencies = zip(*most_common_accents)
    
    # Abbreviations for accents
    accent_abbreviations = {
        "United States English": "US",
        "England English": "UK",
        "Unknown": "Unknown",
        "India and South Asia (India, Pakistan, Sri Lanka)": "S. Asia",
        "Australian English": "AUS",
        "Canadian English": "CAN.",
        "Scottish English": "Scottish",
        "German English,Non native speaker": "GER",
        "Irish English": "Irish",
        "Filipino": "Filipino",
        "Northern Irish": "N. Irish",
    }
    
    abbreviated_categories = [accent_abbreviations[cat] for cat in categories]
    
    plt.bar(abbreviated_categories, frequencies)
    plt.xlabel('Accent')
    plt.ylabel('Frequency')
    plt.title(f'Accent Distribution in {split} split (Top 10)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"traits_distribution_{split}_split.png"))
