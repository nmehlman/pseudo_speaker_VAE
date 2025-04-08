import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torchmetrics import Accuracy
from ps_vae.model import cVAE
from ps_vae.classifier import Classifier
from typing import Any, Dict, Optional, Tuple


class PseudoSpeakerVAE(pl.LightningModule):
    """
    PyTorch Lightning Module for the Pseudo-Speaker Variational Autoencoder (VAE).
    This module supports optional classification and configurable loss weights.
    """

    def __init__(self, **hparams: Dict[str, Any]) -> None:
        """
        Initialize the PseudoSpeakerVAE module.

        Args:
            hparams (dict): Hyperparameters for the model, including model configuration,
                            optimizer settings, and optional classifier configuration.
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes: int = hparams["model"]["num_classes"]
        self.latent_dim: int = hparams["model"]["latent_dim"]

        # Initialize the VAE model
        self.model = cVAE(**hparams["model"])
        self.priors_mean = nn.Parameter(torch.randn(1, self.num_classes, self.latent_dim))
        self.priors_log_sigma = nn.Parameter(torch.randn(1, self.num_classes, self.latent_dim))

        # Load VAE checkpoint if provided
        if hparams.get("vae_checkpoint", None):
            print(f"Using VAE checkpoint {hparams['vae_checkpoint']}")
            self.load_vae_from_checkpoint(hparams["vae_checkpoint"])

        # Optionally freeze the VAE model
        if hparams.get("freeze_vae", False):
            print("Freezing VAE")
            for param in self.model.parameters():
                param.requires_grad = False

        # Initialize the classifier if provided
        if "classifier" in hparams:
            self.classifier = Classifier(**hparams["classifier"])
            self.accuracy = Accuracy(task="multiclass", num_classes=hparams["classifier"]["num_classes"])
        else:
            self.classifier = None

        # Loss weights and options
        self.kl_loss_weight: float = hparams.get("kl_loss_weight", 1.0)
        self.classifier_loss_weight: float = hparams.get("classifier_loss_weight", 1.0)
        self.use_cos_loss: bool = hparams.get("use_cos_loss", False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple containing reconstructed input, latent mean, latent log variance,
            class probabilities, and latent class distribution.
        """
        x_hat, mu, sigma, c, pi = self.model(x)
        return x_hat, mu, sigma, c, pi

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variables into reconstructed input.

        Args:
            z (torch.Tensor): Latent variables.
            c (torch.Tensor): Class probabilities.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        return self.model.decoder(z, c)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.

        Args:
            batch (tuple): Batch of input data and labels.
            batch_idx (int): Batch index.

        Returns:
            dict: Dictionary containing the total loss.
        """
        x, y = batch
        x_hat, mu, log_sigma, c, pi = self(x)

        if self.classifier:
            y_hat = self.classifier(mu)
            classifier_loss = nn.functional.cross_entropy(y_hat, y)
            classifier_acc = self.accuracy(y_hat, y)
            self.log("train_classifier_acc", classifier_acc, sync_dist=True)
        else:
            classifier_loss = 0

        total_loss, recon_loss, kl_loss = self.compute_losses(x, x_hat, mu, log_sigma, pi, classifier_loss)

        # Logging
        self.log("train_loss", total_loss, sync_dist=True)
        self.log("train_recon_loss", recon_loss, sync_dist=True)
        self.log("train_kl_loss", kl_loss, sync_dist=True)
        if self.classifier:
            self.log("train_classifier_loss", classifier_loss, sync_dist=True)

        return {"loss": total_loss}

    def compute_losses(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        pi: torch.Tensor,
        classifier_loss: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the reconstruction, KL divergence, and total loss.

        Args:
            x (torch.Tensor): Original input.
            x_hat (torch.Tensor): Reconstructed input.
            mu (torch.Tensor): Latent mean.
            log_sigma (torch.Tensor): Latent log variance.
            pi (torch.Tensor): Class probabilities.
            classifier_loss (float): Classifier loss.

        Returns:
            Tuple containing total loss, reconstruction loss, and KL divergence loss.
        """
        if self.use_cos_loss:
            recon_loss = nn.functional.cosine_embedding_loss(x_hat, x, torch.ones(x.size(0)).to(x.device))
        else:
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean") / 10  # Roughly normalize the mse loss

        # Expand encoder outputs for broadcasting: [B, 1, D]
        mu = mu.unsqueeze(1)
        log_sigma = log_sigma.unsqueeze(1)

        # [1, K, D] -> broadcast with [B, K, D]
        sigma_q = log_sigma.exp()
        sigma_p = self.priors_log_sigma.exp()

        kl_per_class = 0.5 * torch.sum(
            2 * self.priors_log_sigma - 2 * log_sigma +
            (sigma_q ** 2 + (mu - self.priors_mean) ** 2) / (sigma_p ** 2) - 1,
            dim=2  # sum over latent dim
        )  # shape: [B, K]

        # Expectation over q(c|x) using pi
        kl_loss_1 = torch.sum(pi * kl_per_class, dim=1)  # shape: [B]
        kl_loss_2 = torch.sum(pi * (pi + 1e-8).log(), dim=1) + torch.log(torch.tensor(self.num_classes))

        kl_loss = kl_loss_1.mean() + kl_loss_2.mean()

        total_loss = (
            recon_loss +
            self.kl_loss_weight * kl_loss +
            self.classifier_loss_weight * classifier_loss
        )

        return total_loss, recon_loss, kl_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.

        Args:
            batch (tuple): Batch of input data and labels.
            batch_idx (int): Batch index.

        Returns:
            dict: Dictionary containing the total loss.
        """
        x, y = batch
        x_hat, mu, log_sigma, c, pi = self(x)

        if self.classifier:
            y_hat = self.classifier(mu)
            classifier_loss = nn.functional.cross_entropy(y_hat, y)
            classifier_acc = self.accuracy(y_hat, y)
            self.log("val_classifier_acc", classifier_acc, sync_dist=True)
        else:
            classifier_loss = 0

        total_loss, recon_loss, kl_loss = self.compute_losses(x, x_hat, mu, log_sigma, pi, classifier_loss)

        # Logging
        self.log("val_loss", total_loss, batch_size=x.size(0), sync_dist=True)
        self.log("val_recon_loss", recon_loss, batch_size=x.size(0), sync_dist=True)
        self.log("val_kl_loss", kl_loss, batch_size=x.size(0), sync_dist=True)
        if self.classifier:
            self.log("val_classifier_loss", classifier_loss, batch_size=x.size(0), sync_dist=True)

        return {"loss": total_loss}

    def load_vae_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the VAE model from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        vae_state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        self.model.load_state_dict(vae_state_dict)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer and scheduler configuration.
        """
        optimizer = Adam(self.parameters(), **self.hparams["optimizer"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.hparams["scheduler"])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The learning rate scheduler instance
                'interval': 'epoch',
                'frequency': 1,
            }
        }