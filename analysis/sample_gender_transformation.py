from ps_vae.lightning import PseudoSpeakerVAE
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
from tqdm import tqdm
import os
import shutil

MODEL_CKPT = '/project/shrikann_35/nmehlman/logs/ps_vae/cv_freevc_gender_classifier/version_1/checkpoints/epoch=199-step=29800.ckpt'
CV_ROOT = '/project/shrikann_35/tiantiaf/arts/cv-corpus-11.0-2022-09-21/en/'
SAVE_DIR = '/home1/nmehlman/arts/pseudo_speakers/samples/gender_transformation/example_3_with_prior'
SOURCE_GENDER = 'male'
TARGET_GENDER = 'female'
SOURCE_ACCENT = 'United States English'
EMBED_DIR = os.path.join(CV_ROOT, 'embeds_vc/train/')
AUDIO_DIR = os.path.join(CV_ROOT, 'clips')
MAX_STEPS = 10000
STEP_SIZE = 0.02
NOISE_WEIGHT = 0.0
PRIOR_WEIGHT = 0.5
SAVE_INTERVAL = 50
THRESHOLD = 0.95

GENDER_MAPPING = {
        'male': 0,
        'female': 1,
        'other': 2 
}

if __name__ == "__main__":

    # Select sample at random
    metadata = pd.read_csv(os.path.join(CV_ROOT, 'train_embeds.tsv'), sep='\t')

    if SOURCE_ACCENT is not None:
        metadata_filtered = metadata[ # Filter by demographics
            (metadata['gender'] == SOURCE_GENDER) 
            * (metadata['accents'] == SOURCE_ACCENT)
        ]
    else:
        metadata_filtered  = metadata[ # Filter by demographics
            (metadata['gender'] == SOURCE_GENDER)
        ]

    print(f"Found {len(metadata_filtered)} valid samples")

    # Select and load random sample
    sample_idx = torch.randint(0, len(metadata_filtered), (1,)).item()
    sample_row = metadata_filtered.iloc[sample_idx]
    print(f"Selected sample {sample_row['path'].replace('.mp3', '')}")

    embed_path = os.path.join(EMBED_DIR, sample_row['path'].replace('.mp3', '.pth'))
    emebd = torch.load(embed_path)

    # Load the model
    vae_model = PseudoSpeakerVAE.load_from_checkpoint(MODEL_CKPT)
    vae_model.eval()

    # Get original latent vector
    with torch.no_grad():
        _, z, _ = vae_model(emebd.unsqueeze(0))
        z.requires_grad_()

    classifier_target = GENDER_MAPPING[TARGET_GENDER]
    pbar = tqdm(range(MAX_STEPS), desc="Modifying sample")
    history = []
    for step in pbar:
    
        # Get predicted probabilities from classifier
        logits = vae_model.classifier(z)
        log_probs = F.log_softmax(logits, dim=-1)
        
        log_p_y_given_z = log_probs[:, classifier_target]
        
        log_p_z = -0.5 * (z ** 2).sum(dim=1)
        
        log_p_z_given_y = log_p_y_given_z + PRIOR_WEIGHT * log_p_z
        grad = torch.autograd.grad(log_p_z_given_y.sum(), z)[0] # Get gradient
                    
        noise = torch.randn_like(z)

        p_y_given_z = log_p_y_given_z.clone().exp()
        if step % SAVE_INTERVAL == 0:
            history.append(
                {   
                    "step": step,
                    "classifier_prob": p_y_given_z.item(),
                    "latent": z.detach().squeeze().cpu()
                }
            )
        
        z = z + 0.5 * (STEP_SIZE ** 2) * grad + STEP_SIZE * NOISE_WEIGHT * noise # Gradient ascent step

        z.requires_grad_()
        pbar.set_postfix({"Classifier Prob": p_y_given_z.item()})

        if p_y_given_z.item() > THRESHOLD:
            break

    print(f"Converged after {step} steps")
    
    # Save the history + embeddings
    pickle.dump(history, open(os.path.join(SAVE_DIR, 'history.pkl'), "wb"))
    
    embedding_save_dir = os.path.join(SAVE_DIR, "embeds")
   
    os.makedirs(embedding_save_dir, exist_ok=True)
    for i, sample in enumerate(history):
        x_hat = vae_model.decode(sample['latent'].unsqueeze(0))
        x_hat = x_hat.squeeze().cpu()
        torch.save(
            x_hat,
            os.path.join(embedding_save_dir, f"embed_{i}_step_{sample['step']}_prob_{sample['classifier_prob']:.4f}.pt")
        )
    
    # Save the souce audio
    source_audio_path = os.path.join(AUDIO_DIR, sample_row['path'])
    shutil.copy(
        source_audio_path,
        os.path.join(SAVE_DIR, f"source_{sample_row['path']}")
    )
