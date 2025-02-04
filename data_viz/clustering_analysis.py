import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from ps_vae.data import CVEmbeddingDataset
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from plot_embeddings import stratified_sampling

dim_reduction_methods = {
    "PCA": PCA,
    "TSNE": TSNE,
    "MDS": MDS,
    "Isomap": Isomap
}

clustering_methods = { 
    "KMeans": KMeans,
    "DBSCAN": DBSCAN,
    "AgglomerativeClustering": AgglomerativeClustering
}

metadata_mapping = {
    "age": {
        "teens": 1,
        "twenties": 2,
        "thirties": 3,
        "fourties": 4,
        "fifties": 5,
        "sixties": 6,
        "seventies": 7,
        "eighties": 8,
        "nineties": 9,
    },
    "gender": {
        "male": 0,
        "female": 1,
        "other": 2
    }
}

if __name__ == "__main__":

    N_SAMPLES = 1000
    DIM_REDUCTION = "MDS"
    CLUSTERING = "KMeans"
    N_DIMS = 2
    N_CLUSTERS = 10
    
    filter_fcn = lambda sample: sample["accents"] == "United States English"
    
    # Load data
    data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    save_dir = "/home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/plots/"
    dataset = CVEmbeddingDataset(data_root, split="train")
    
    # Select samples at random
    sample_idx = stratified_sampling(dataset, N_SAMPLES, filter_fcn=filter_fcn)
    
    # Get embeddings
    samples = np.stack([dataset[i][0].numpy() for i in sample_idx], axis=0)
    
    # Compute metadata features
    metadata = [dataset[i][1] for i in sample_idx]
    metadata_feats = []
    for meta in metadata:
        sample_feats = []
        for meta_name, meta_mapping in metadata_mapping.items():
            sample_feats.append(meta_mapping[meta[meta_name]])
        metadata_feats.append(sample_feats)
        
    metadata_feats = np.array(metadata_feats) # shape: (N_SAMPLES, N_METADATA_FEATS)
    
    
    # Perform dimensionality reduction
    dim_redox = dim_reduction_methods[DIM_REDUCTION](n_components=N_DIMS)
    samples_dim_redox = dim_redox.fit_transform(samples)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    clusters = kmeans.fit_predict(samples_dim_redox)
    
    # Plot
    dim_redox_plot = dim_reduction_methods[DIM_REDUCTION](n_components=2)
    samples_plot = dim_redox_plot.fit_transform(samples)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(samples_plot[:, 0], samples_plot[:, 1], c=clusters, cmap="tab10")
    
    # Create a legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    plt.savefig(os.path.join(save_dir, f"clustering.png"))
    
    # Analyze clusters via metadata
    for n in range(N_CLUSTERS):
        cluster_idx = np.where(clusters == n)[0]
        cluster_metadata = metadata_feats[cluster_idx]
        cov = np.cov(cluster_metadata.T)
        
        print(f"Cluster {n} metadata covariance matrix:")
        print(cov)
    
    
