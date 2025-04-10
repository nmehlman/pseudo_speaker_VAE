import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from ps_vae.data.cv import CVEmbeddingDataset
from ps_vae.lightning import PseudoSpeakerVAE
from typing import Callable, List, Optional

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

def stratified_sampling(
    dataset: CVEmbeddingDataset, 
    n_samples: int, 
    stratification_fcn: Optional[Callable[[dict], str]] = None, 
    filter_fcn: Optional[Callable[[dict], bool]] = None
) -> List[int]:
    """
    Perform stratified sampling on the dataset.

    Args:
        dataset (CVEmbeddingDataset): The dataset to sample from.
        n_samples (int): The number of samples to draw.
        stratification_fcn (Optional[Callable[[dict], str]]): Function to determine the stratification group of a sample.
        filter_fcn (Optional[Callable[[dict], bool]]): Function to filter valid samples.

    Returns:
        List[int]: List of indices of the sampled data points.
    """
    metadata = dataset.metadata
    files = dataset.embedding_files
    
    if filter_fcn is None:
        valid_samples = list(range(len(files)))
    else:
        valid_samples = [i for i in range(len(files)) if filter_fcn(metadata[files[i]])]

    if stratification_fcn is None:
        return np.random.choice(valid_samples, n_samples, replace=False)
    else:
        stratification_valid = [stratification_fcn(metadata[files[i]]) for i in valid_samples]
        unique_stratifications = list(set(stratification_valid))

        strat_samples = []
        for strat in unique_stratifications:
            strat_idxs_valid = [i for i in range(len(stratification_valid)) if stratification_valid[i] == strat]
            group_samples_valid = np.random.choice(strat_idxs_valid, n_samples // len(unique_stratifications), replace=False)
            strat_samples.extend([valid_samples[i] for i in group_samples_valid])

        return strat_samples

if __name__ == "__main__":
    
    N_SAMPLES = 1000
    DEMOGRAPHIC = "gender"
    APPLY_AGE_GROUPINGS = True
    MODEL_CKPT = '/project/shrikann_35/nmehlman/logs/ps_vae/cv_freevc_gender_classifier/version_2/checkpoints/epoch=199-step=59400.ckpt'
    DATA_ROOT = "/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/"
    SE_MODEL = "vc"
    SAVE_AS = "/home1/nmehlman/arts/pseudo_speakers/pseudo_speaker_VAE/plots/vae_latent_gender+classifier.png"
    USE_TSNE = False  # Set to True to use TSNE, False to use PCA
    
    filter_fcn = None #lambda sample: sample['accents'] == 'United States English'
    stratification_fcn = None
    
    dataset = CVEmbeddingDataset(DATA_ROOT, split="train", se_model=SE_MODEL)

    if MODEL_CKPT:
        model = PseudoSpeakerVAE.load_from_checkpoint(MODEL_CKPT)
        model.eval()
    else:
        model = None

    sample_idx = stratified_sampling(dataset, N_SAMPLES, stratification_fcn=stratification_fcn, filter_fcn=filter_fcn)
    
    if model:
        samples = []
        for i in sample_idx:
            _, mu, _ = model(dataset[i][0].unsqueeze(0))
            samples.append(mu.detach().squeeze().numpy())
        samples = np.stack(samples, axis=0)
        
    else:
        samples = np.stack([dataset[i][0].numpy() for i in sample_idx], axis=0)

    demos = [dataset[i][1][DEMOGRAPHIC] for i in sample_idx]
    demos = [demo if demo else "Unknown" for demo in demos]

    if APPLY_AGE_GROUPINGS and DEMOGRAPHIC == "age":
        demos = [AGE_GROUPINGS[demo] for demo in demos]
    
    unique_demos = sorted(list(set(demos)))
    demo_to_int = {demo: idx for idx, demo in enumerate(unique_demos)}
    demos_int = [demo_to_int[demo] for demo in demos]

    if USE_TSNE:
        reducer = TSNE(n_components=2, random_state=0)
        samples_proj = reducer.fit_transform(samples)
        xlabel, ylabel = "TSNE 1", "TSNE 2"
    else:
        reducer = PCA(n_components=2, random_state=0)
        samples_proj = reducer.fit_transform(samples)
        xlabel, ylabel = "PCA 1", "PCA 2"

    scatter = plt.scatter(
        samples_proj[:, 0], samples_proj[:, 1], c=demos_int, cmap="tab10", marker="."
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("{} of {} colored by {}".format("TSNE" if USE_TSNE else "PCA", "latent vectors" if model else "raw embeddings", DEMOGRAPHIC))

    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [unique_demos[idx] for idx in range(len(unique_demos))]
    plt.legend(handles, legend_labels, title="Demographics")

    plt.savefig(SAVE_AS)
