import pytorch_lightning as pl
import torch.nn as nn
from ps_vae.model import VAEModel
import torch
from torch.optim import Adam


class PseudoSpeakerVAE(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()

        self.save_hyperparameters()

        self.model = VAEModel(**hparams["model"])

        self.kl_loss_weight = hparams.get("kl_loss_weight", 1.0)

    def forward(self, x):
        x_hat, mu, sigma = self.model(x)
        return x_hat, mu, sigma

    def training_step(self, batch: tuple, batch_idx: int) -> float:

        x, _ = batch
        x_hat, mu, log_sigma = self(x)

        mse_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1)
        )

        total_loss = mse_loss + self.kl_loss_weight * kl_loss
        
        # Logging
        self.log("train_loss", total_loss)
        self.log("train_mse_loss", mse_loss)
        self.log("train_kl_loss", kl_loss)

        return {"loss": total_loss}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), **self.hparams["optimizer"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.hparams["scheduler"])
        return {
            'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,  # The learning rate scheduler instance
                    'interval': 'epoch',       
                    'frequency': 1,        
                }
            }