from ps_vae.lightning import PseudoSpeakerVAE
import torch
import os
from torch import Tensor
from tqdm import tqdm
import torch.nn.functional as F

def unconditional_synthesis(vae_model: PseudoSpeakerVAE, num_samples: int) -> Tensor:
    """
    Generate unconditioned samples using the provided VAE model.
    
    Args:
        vae_model (PseudoSpeakerVAE): The pre-trained VAE model.
        num_samples (int): Number of samples to generate.
    
    Returns:
        list: Generated samples.
    """
    dim = vae_model.hparams.model['latent_dim']
    z = torch.randn((num_samples, dim))
    
    x_hat = vae_model.decode(z).detach().cpu()
    
    return x_hat

def conditional_synthesis(
    vae_model: PseudoSpeakerVAE, 
    num_samples: int,
    classifier_target: int,
    step_size: float = 0.01,
    num_steps: int = 100,
    noise_weight: float = 1.0,
    binary: bool = False,
    return_history: bool = False
    ) -> Tensor:
    """
    Generate conditioned samples using the provided VAE model.
    
    Args:
        vae_model (PseudoSpeakerVAE): The pre-trained VAE model.
        num_samples (int): Number of samples to generate.
        classifier_target (int): Target class for the classifier.
        step_size (float, optional): Step size for the gradient ascent. Default is 0.01.
        num_steps (int, optional): Number of steps for the gradient ascent. Default is 100.
        binary (bool, optional): Whether the classifier is binary. Default is False.
        return_history (bool, optional): Return history of z iterations. Default is False.
    
    Returns:
        Tensor: Generated samples.
    """
    
    dim = vae_model.hparams.model['latent_dim']
    z = torch.randn((num_samples, dim), requires_grad=True)
    
    pbar = tqdm(range(num_steps), desc="Generating samples")
    history = []
    for _ in pbar:
    
        # Get predicted probabilities from classifier
        logits = vae_model.classifier(z)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute log p(y|z) + log p(z)
        if binary:
            if classifier_target == 1:
                log_p_y_given_z = log_probs
            elif classifier_target == 0:
                log_p_y_given_z = 1-log_probs
            else:
                raise ValueError("classifier_target must be 0 or 1 for binary classifier")
        else:
            log_p_y_given_z = log_probs[:, classifier_target]
            
        log_p_z = -0.5 * (z ** 2).sum(dim=1)
        
        log_p_z_given_y = log_p_y_given_z + log_p_z
        grad = torch.autograd.grad(log_p_z_given_y.sum(), z)[0] # Get gradient
        
        noise = torch.randn_like(z)
        
        z = z + 0.5 * (step_size ** 2) * grad + step_size * noise_weight * noise
        
        history.append(z.detach().cpu().numpy())
        
        z.requires_grad_()
        
        pbar.set_postfix({"Log Likelihood": log_p_z_given_y.mean().item()})
    
    x_hat = vae_model.decode(z).detach().cpu()
    
    if return_history:  
        return x_hat, history   
    else:
        return x_hat
    

if __name__ == "__main__":
    
    VAE_CKPT_PATH = "/project/shrikann_35/nmehlman/logs/ps_vae/cv_freevc_01/version_1/checkpoints/epoch=199-step=29800.ckpt"
    N_SAMPLES = 16
    SAVE_DIR = "/home1/nmehlman/arts/pseudo_speakers/generated_embeddings"
    
    model = PseudoSpeakerVAE.load_from_checkpoint(VAE_CKPT_PATH)
    
    x_hat = unconditional_synthesis(
        model, 
        N_SAMPLES,
    )
    
    
    for i, x in enumerate(x_hat):
        torch.save(x, os.path.join(SAVE_DIR, f"sample_{i}.pt"))        