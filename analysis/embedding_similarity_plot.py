import os
import torch
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import argparse

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compute and plot pair-wise cosine similarity of .pt vectors.")
    parser.add_argument("directory", type=str, help="Directory containing the .pt files.")
    args = parser.parse_args()

    # Directory containing the .pt files
    directory = args.directory

    # Load all .pt vectors
    vectors = []
    file_names = []
    embedding_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    for file_name in sorted(embedding_files, key=lambda x: int(x.split('_')[1].replace('.pt', ''))):
        if file_name.endswith('.pt'):
            file_path = os.path.join(directory, file_name)
            vector = torch.load(file_path).squeeze()
            vectors.append(vector.detach().numpy())
            file_names.append(file_name)

    # Compute pair-wise cosine similarity
    cosine_sim_matrix = cosine_similarity(vectors)

    # Plot the cosine similarity matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cosine_sim_matrix, xticklabels=file_names, yticklabels=file_names, annot=True, cmap='coolwarm')
    plt.title('Pair-wise Cosine Similarity')
    
    # Save
    output_path = os.path.join(directory, 'cosine_similarity_matrix.png')
    plt.savefig(output_path)