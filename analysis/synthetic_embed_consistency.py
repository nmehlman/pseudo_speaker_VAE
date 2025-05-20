import torch
import os
from ps_vae.utils import map_cv_gender_to_label, map_cv_age_to_label, sample_from_cv_metadata
from ps_vae.inference import conditional_synthesis
from ps_vae.lightning import PseudoSpeakerVAE
import shutil

import pandas as pd
from itertools import product
import torchaudio
import tqdm
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate synthetic embeddings for consistency analysis.")
    parser.add_argument("--demo", type=str, default="gender", help="Demographic attribute to test (e.g., 'gender').")
    parser.add_argument("--demos_to_test", nargs="+", default=["male", "female"], help="List of demographic values to test.")
    parser.add_argument("--n_samples_per_demo", type=int, default=5, help="Number of samples to generate per demographic value.")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to the VAE model checkpoint.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated samples.")
    parser.add_argument("--step_size", type=float, default=0.02, help="Step size for conditional synthesis.")
    parser.add_argument("--num_steps", type=int, default=150, help="Number of steps for conditional synthesis.")
    parser.add_argument("--noise_weight", type=float, default=0.5, help="Noise weight for conditional synthesis.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on ('cuda' or 'cpu').")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load the VAE model
    model = PseudoSpeakerVAE.load_from_checkpoint(args.vae_ckpt_path)
    model = model.to(device)

    # Generate synthetic target embedding for each demographic value
    for demo in args.demos_to_test:
        print(f"Generating synthetic target embeddings for {demo}...")
        subdir = os.path.join(args.save_dir, "embeds", f"{demo}")
        os.makedirs(subdir, exist_ok=True)

        # Generate synthetic embeddings using conditional synthesis
        classifier_target = map_cv_gender_to_label(demo)  # Map demo to classifier target

        synthetic_samples = conditional_synthesis(
            vae_model=model,
            num_samples=args.n_samples_per_demo,
            classifier_target=classifier_target,
            step_size=args.step_size,
            num_steps=args.num_steps,
            noise_weight=args.noise_weight,
            device=device
        )

        # Save generated samples
        for i, sample in enumerate(synthetic_samples):
            torch.save(sample, os.path.join(subdir, f"sample_{i}.pt"))

    # Sample random source speeches from each demographic value
    metadata_path = os.path.join(args.data_root, "train_embeds.tsv")
    metadata = pd.read_csv(metadata_path, sep="\t")
    for demo in args.demos_to_test:
        subdir = os.path.join(args.save_dir, "speech", f"{demo}")
        os.makedirs(subdir, exist_ok=True)

        # Sample random source speeches
        sources = sample_from_cv_metadata(
            metadata, args.n_samples_per_demo, conditions={args.demo: demo}
        )

        for filename in sources['path']:
            full_audio_path = os.path.join(args.data_root, 'clips', filename)
            save_path = os.path.join(subdir, filename)
            shutil.copy(full_audio_path, save_path)