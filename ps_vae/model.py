from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F

# TODO - Add tau scheduling


def sample_gumbel(shape: torch.Size, eps: float = 1e-20) -> torch.Tensor:
    """
    Samples from the standard Gumbel(0, 1) distribution.

    Args:
        shape (torch.Size): Shape of the sampled tensor.
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Sampled Gumbel noise.
    """
    uniform_noise = torch.rand(shape)
    return -torch.log(-torch.log(uniform_noise + eps) + eps)


def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Draws a sample from the Gumbel-Softmax distribution.

    Args:
        logits (torch.Tensor): Unnormalized logits of shape [batch_size, num_categories].
        tau (float): Temperature parameter controlling approximation (lower = closer to categorical).

    Returns:
        torch.Tensor: Gumbel-Softmax sample of shape [batch_size, num_categories] (differentiable).
    """
    gumbel_noise = sample_gumbel(logits.shape)
    perturbed_logits = logits + gumbel_noise
    return F.softmax(perturbed_logits / tau, dim=-1)


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Gumbel-Softmax function, optionally discretized (hard one-hot) using the straight-through trick.

    Args:
        logits (torch.Tensor): Unnormalized logits of shape [batch_size, num_categories].
        tau (float): Temperature parameter.
        hard (bool): If True, discretize output using the straight-through trick (one-hot vector).

    Returns:
        torch.Tensor: Gumbel-Softmax sample of shape [batch_size, num_categories].
    """
    soft_sample = gumbel_softmax_sample(logits, tau)

    if hard:
        # Straight-through trick: discretize the soft sample
        max_indices = soft_sample.max(dim=-1, keepdim=True)[1]
        hard_sample = torch.zeros_like(logits).scatter_(-1, max_indices, 1.0)

        # Maintain gradients using the straight-through trick
        return (hard_sample - soft_sample).detach() + soft_sample
    return soft_sample


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        encoder_dim: int = 256,
    ):
        """
        Encoder network for the class posterior.

        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space.
            num_classes (int): Number of classes in the dataset.
            hidden_dim (int): Dimension of the hidden layers.
            encoder_dim (int): Dimension of the encoder output.
        """
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder_dim),
            nn.ReLU(),
        )

        self.class_logits_layer = nn.Linear(encoder_dim, num_classes)
        self.latent_mean_layer = nn.Linear(encoder_dim + num_classes, latent_dim)
        self.latent_log_std_layer = nn.Linear(encoder_dim + num_classes, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Class probabilities (pi) of shape (batch_size, num_classes).
                - Class logits of shape (batch_size, num_classes).
                - Latent mean (mu) of shape (batch_size, latent_dim).
                - Latent standard deviation (sigma) of shape (batch_size, latent_dim).
        """
        encoded_features = self.encoder(x)
        class_logits = self.class_logits_layer(encoded_features)
        class_probs = class_logits.softmax(dim=1)

        combined_features = torch.cat([encoded_features, class_probs], dim=1)
        latent_mean = self.latent_mean_layer(combined_features)
        latent_log_std = self.latent_log_std_layer(combined_features)

        return class_probs, class_logits, latent_mean, latent_log_std


class Decoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_classes: int, hidden_dim: int = 256):
        """
        Decoder network for reconstructing the input.

        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space.
            num_classes (int): Number of classes in the dataset.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, latent_z: torch.Tensor, class_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder network.

        Args:
            latent_z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
            class_probs (torch.Tensor): Class probabilities tensor of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Reconstructed input tensor of shape (batch_size, input_dim).
        """
        combined_input = torch.cat([latent_z, class_probs], dim=1)
        reconstructed_x = self.decoder(combined_input)
        return reconstructed_x
    
class cVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        tau: float = 1.0,
    ):
        """
        Conditional Variational Autoencoder (cVAE) model.

        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space.
            num_classes (int): Number of classes in the dataset.
            hidden_dim (int): Dimension of the hidden layers.
            tau (float): Temperature parameter for Gumbel-Softmax.
        """
        super(cVAE, self).__init__()

        self.encoder = Encoder(input_dim, latent_dim, num_classes, hidden_dim)
        self.decoder = Decoder(input_dim, latent_dim, num_classes, hidden_dim)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the cVAE model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Reconstructed input (x_recon) of shape (batch_size, input_dim).
                - Latent mean (mu) of shape (batch_size, latent_dim).
                - Latent standard deviation (sigma) of shape (batch_size, latent_dim).
                - Soft sampled class (c) of shape (batch_size, num_classes).
                - Class probabilities (pi) of shape (batch_size, num_classes).
        """
        class_probs, class_logits, latent_mean, latent_log_std = self.encoder(x)
        sampled_class = gumbel_softmax(class_logits, tau=self.tau, hard=False)
        latent_z = latent_mean + latent_log_std.exp() * torch.randn_like(latent_mean)
        reconstructed_x = self.decoder(latent_z, sampled_class)

        return reconstructed_x, latent_mean, latent_log_std, sampled_class, class_probs


if __name__ == "__main__":
    
    input_dim = 784 
    latent_dim = 20
    num_classes = 10
    batch_size = 16

    model = cVAE(input_dim, latent_dim, num_classes)
    x_dummy = torch.randn(batch_size, input_dim)

    x_recon, mu, sigma, c, pi = model(x_dummy)

    print("x_recon shape:", x_recon.shape)  # expect: [batch_size, input_dim]
    print("mu shape:", mu.shape)            # expect: [batch_size, latent_dim]
    print("sigma shape:", sigma.shape)      # expect: [batch_size, latent_dim]
    print("c shape:", c.shape)              # expect: [batch_size, num_classes]
    print("pi shape:", pi.shape)            # expect: [batch_size, num_classes]
