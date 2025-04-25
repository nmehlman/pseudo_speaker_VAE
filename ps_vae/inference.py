from ps_vae.lightning import PseudoSpeakerVAE
import torch
import os
from torch import Tensor
from tqdm import tqdm
import json
import argparse
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
    classifier_target: int | dict,
    step_size: float = 0.01,
    num_steps: int = 100,
    noise_weight: float = 1.0,
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

    def _get_classifer_probs(logits, classifier_target):
        log_probs = F.log_softmax(logits, dim=-1)
            
        # Compute log p(y|z) + log p(z)
        if log_probs.shape[1] == 1: # Binary
            if classifier_target == 1:
                log_p_y_given_z = log_probs
            elif classifier_target == 0:
                log_p_y_given_z = 1-log_probs
            else:
                raise ValueError("classifier_target must be 0 or 1 for binary classifier")
        else:
            log_p_y_given_z = log_probs[:, classifier_target]

        return log_p_y_given_z
    
    dim = vae_model.hparams.model['latent_dim']
    z = torch.randn((num_samples, dim), requires_grad=True)
    
    pbar = tqdm(range(num_steps), desc="Generating samples")
    history = []
    for _ in pbar:
    
        # Get predicted probabilities from classifier
        logits = vae_model.classifier(z)
        if isinstance(logits, dict):
            assert isinstance(classifier_target, dict), "classifier_target must be a dict for multi-label classifier"
            log_p_y_given_z = 0
            for label, target in classifier_target.items():
                log_p_y_given_z += _get_classifer_probs(logits[label], target)

        else:
            log_p_y_given_z = _get_classifer_probs(logits, classifier_target)
            
        log_p_z = -0.5 * (z ** 2).sum(dim=1)
        
        log_p_z_given_y = log_p_y_given_z + log_p_z
        grad = torch.autograd.grad(log_p_z_given_y.sum(), z)[0] # Get gradient
        
        noise = torch.randn_like(z)
        
        z = z + 0.5 * (step_size ** 2) * grad + step_size * noise_weight * noise
        
        history.append(z.detach().cpu().numpy())
        
        z.requires_grad_()
        
        pbar.set_postfix({f"Log Likelihood": log_p_z_given_y.mean().item(), "Classifier Prob": log_p_y_given_z.exp().mean().item()})
    
    x_hat = vae_model.decode(z).detach().cpu()
    
    if return_history:  
        return x_hat, history   
    else:
        return x_hat
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate synthetic embeddings using a VAE model.")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to the VAE checkpoint.")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated samples.")
    parser.add_argument("--synthesis_type", type=str, choices=["conditional", "unconditional"], required=True, help="Type of synthesis: 'conditional' or 'unconditional'.")
    parser.add_argument("--classifier_target", type=str, default='1', help="Target class for the classifier (only for conditional synthesis).")
    parser.add_argument("--num_steps", type=int, default=5000, help="Number of gradient ascent steps (only for conditional synthesis).")
    parser.add_argument("--step_size", type=float, default=0.01, help="Step size for gradient ascent (only for conditional synthesis).")
    parser.add_argument("--noise_weight", type=float, default=1.0, help="Weight of the noise added during synthesis (only for conditional synthesis).")

    args = parser.parse_args()

    # Parse classifier_target as int or dict
    try:
        classifier_target = json.loads(args.classifier_target)
    except json.JSONDecodeError:
        classifier_target = int(args.classifier_target)

    model = PseudoSpeakerVAE.load_from_checkpoint(args.vae_ckpt_path)

    if args.synthesis_type == "conditional":
        x_hat = conditional_synthesis(
            model, 
            classifier_target=classifier_target,
            num_samples=args.n_samples,
            num_steps=args.num_steps,
            step_size=args.step_size,
            noise_weight=args.noise_weight
        )
    elif args.synthesis_type == "unconditional":
        x_hat = unconditional_synthesis(
            model,
            num_samples=args.n_samples
        )

    os.makedirs(args.save_dir, exist_ok=True)
    for i, x in enumerate(x_hat):
        torch.save(x, os.path.join(args.save_dir, f"sample_{i}.pt"))