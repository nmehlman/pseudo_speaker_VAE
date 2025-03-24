from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F


class VAEModel(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 64, normalize_decoder: bool = False):
        
        super().__init__()

        self.normalize_decoder = normalize_decoder

        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

        self.encoder_sigma = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - x_hat (torch.Tensor): Reconstructed input tensor of shape (batch_size, input_dim).
                - mu (torch.Tensor): Mean of the latent space distribution of shape (batch_size, latent_dim).
                - sigma (torch.Tensor): Standard deviation of the latent space distribution of shape (batch_size, latent_dim).
        """

        mu = self.encoder_mu(x)
        log_sigma = self.encoder_sigma(x)
        sigma = torch.exp(0.5 * log_sigma)
        z = mu + sigma * torch.randn_like(sigma)
        x_hat = self.decoder(z)

        if self.normalize_decoder:
            x_hat = F.normalize(x_hat, p=2, dim=1)

        return x_hat, mu, log_sigma
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)
        if self.normalize_decoder:
            x_hat = F.normalize(x_hat, p=2, dim=1)


if __name__ == "__main__":
    model = VAEModel(784, 20)
    z = torch.randn(32, 784)
    x_hat, mu, sigma = model(z)
    print(x_hat.shape, mu.shape, sigma.shape)
