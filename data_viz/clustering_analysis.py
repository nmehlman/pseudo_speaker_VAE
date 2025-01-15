import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from ps_vae.data import CVEmbeddingDataset

AGE_GROUPINGS = {
    "teens": "Young",
    "twenties": "Young",
    "thirties": "Young",
    "fourties": "Middle-aged",
    "fifties": "Middle-aged",
    "sixties": "Middle-aged",
    "seventies": "Old",
    "eighties": "Old",
    "nineties": "Old",
}

if __name__ == "__main__":

    N_SAMPLES = 1000
    DEMOGRAPHIC = "gender"
    APPLY_AGE_GROUPINGS = True
    filter_fcn = lambda sample: sample['accents'] == 'United States English' and AGE_GROUPINGS[sample['age']] == 'Old'
    stratification_fcn = None#lambda sample: AGE_GROUPINGS[sample['age']]

    data_root = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    save_dir = "/home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/plots/"
    dataset = CVEmbeddingDataset(data_root, split="train")

    sample_idx = stratified_sampling(dataset, N_SAMPLES, stratification_fcn=stratification_fcn, filter_fcn=filter_fcn)
    
    samples = np.stack([dataset[i][0].numpy() for i in sample_idx], axis=0)

    # Get demographic info and convert to integers
    demos = [dataset[i][1][DEMOGRAPHIC] for i in sample_idx]
    
    demos = [demo if demo else "Unknown" for demo in demos]  # Formatting
    if APPLY_AGE_GROUPINGS and DEMOGRAPHIC == "age":
        demos = [AGE_GROUPINGS[demo] for demo in demos]
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

    plt.savefig(os.path.join(save_dir,f"raw_embeddings_{DEMOGRAPHIC}_tsne_us_only_old.png"))
