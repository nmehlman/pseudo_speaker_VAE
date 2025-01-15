import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from ps_vae.data import CVEmbeddingDataset

if __name__ == "__main__":

    N_SAMPLES = 1000
    DEMOGRAPHIC = "accents"

    data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    save_dir = "/home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/plots/"
    dataset = CVEmbeddingDataset(data_root, split="train")

    # Randomly sample N_SAMPLES from the dataset
    sample_idx = np.random.choice(len(dataset), N_SAMPLES, replace=False)
    samples = np.stack([dataset[i][0].numpy() for i in sample_idx], axis=0)

    # Get demographic info and convert to integers
    demos = [dataset[i][1][DEMOGRAPHIC] for i in sample_idx]
    demos = [demo if demo else "Unknown" for demo in demos]  # Formatting
    unique_demos = list(set(demos))
    demo_to_int = {demo: idx for idx, demo in enumerate(unique_demos)}
    demos_int = [demo_to_int[demo] for demo in demos]

    # Run TSNE
    tsne = TSNE(n_components=2, random_state=0)
    samples_proj = tsne.fit_transform(samples)

    # Make plot
    scatter = plt.scatter(
        samples_proj[:, 0], samples_proj[:, 1], c=demos_int, cmap="tab10", marker="."
    )
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.title("TSNE of raw embeddings colored by {}".format(DEMOGRAPHIC))

    # Create a legend
    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [unique_demos[idx] for idx in range(len(unique_demos))]
    plt.legend(handles, legend_labels, title="Demographics")

    plt.savefig(os.path.join(save_dir,f"raw_embeddings_{DEMOGRAPHIC}_tsne.png"))
