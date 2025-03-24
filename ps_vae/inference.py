from ps_vae.lightning import PseudoSpeakerVAE
import torch
import os
from torch import Tensor

def unconditional_synthesis(vae_model: PseudoSpeakerVAE, num_samples: int) -> Tensor:
    """
    Generate unconditioned samples using the provided VAE model.
    
    Args:
        vae_model (PseudoSpeakerVAE): The pre-trained VAE model.
        num_samples (int): Number of samples to generate.
    
    Returns:
        list: Generated samples.
    """
    
    z = torch.randn((num_samples, vae_model.hparams.model['latent_dim']))
    
    x_hat = vae_model.decode(z).detach().cpu()
    
    return x_hat
    

if __name__ == "__main__":
    
    VAE_CKPT_PATH = "/project/shrikann_35/nmehlman/logs/ps_vae/vctk_train_01/version_1/checkpoints/epoch=499-step=5000.ckpt"
    N_SAMPLES = 16
    SAVE_DIR = "/home1/nmehlman/arts/pseudo_speakers/generated_embeddings"
    
    model = PseudoSpeakerVAE.load_from_checkpoint(VAE_CKPT_PATH)
    
    x_hat = unconditional_synthesis(model, N_SAMPLES)
    
    for i, x in enumerate(x_hat):
        torch.save(x, os.path.join(SAVE_DIR, f"sample_{i}.pt"))        