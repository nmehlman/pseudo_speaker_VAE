import pytorch_lightning as pl
import torch.nn as nn
from ps_vae.model import VAEModel
from ps_vae.classifier import Classifier
import torch
from torch.optim import Adam
from torchmetrics import Accuracy

class PseudoSpeakerVAE(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()

        self.save_hyperparameters()

        self.model = VAEModel(**hparams["model"])

        if hparams.get("vae_checkpoint", None):
            print(f"Using VAE checkpoint {hparams['vae_checkpoint']}")
            self.load_vae_from_checkpoint(hparams["vae_checkpoint"])

        if hparams.get("freeze_vae", False):
            print("Freezing VAE")
            for param in self.model.parameters():
                param.requires_grad = False

        if 'classifier' in hparams:
            self.classifier = Classifier(**hparams["classifier"])
            self.accuracy = Accuracy(task="multiclass", num_classes=hparams["classifier"]["num_classes"])
        else:
            self.classifier = None
        
        self.kl_loss_weight = hparams.get("kl_loss_weight", 1.0)
        self.classifier_loss_weight = hparams.get("classifier_loss_weight", 1.0)

    def forward(self, x):
        x_hat, mu, sigma = self.model(x)
        return x_hat, mu, sigma

    def training_step(self, batch: tuple, batch_idx: int) -> float:

        x, y = batch
        x_hat, mu, log_sigma = self(x)

        if self.classifier:
            y_hat = self.classifier(mu)
            classifier_loss = nn.functional.cross_entropy(y_hat, y)
            classifier_acc = self.accuracy(y_hat, y)
            self.log("train_classifier_acc", classifier_acc, sync_dist=True)
            self.log("train_classifier_loss", classifier_loss)
        else:
            classifier_loss = 0

        mse_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1)
        )

        total_loss = (
            mse_loss/10 + # Roughly normalize the mse loss
            self.kl_loss_weight * kl_loss + 
            self.classifier_loss_weight * classifier_loss
        )
        
        # Logging
        self.log("train_loss", total_loss)
        self.log("train_mse_loss", mse_loss)
        self.log("train_kl_loss", kl_loss)

        return {"loss": total_loss}
    
    def validation_step(self, batch: tuple, batch_idx: int) -> float:

        x, y = batch
        x_hat, mu, log_sigma = self(x)

        if self.classifier:
            y_hat = self.classifier(mu)
            classifier_loss = nn.functional.cross_entropy(y_hat, y)
            classifier_acc = self.accuracy(y_hat, y)
            self.log("val_classifier_acc", classifier_acc, sync_dist=True)
            self.log("val_classifier_loss", classifier_loss, sync_dist=True)
        else:
            classifier_loss = 0

        mse_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1)
        )

        total_loss = (
            mse_loss/10 + # Roughly normalize the mse loss 
            self.kl_loss_weight * kl_loss + 
            self.classifier_loss_weight * classifier_loss
        )
        
        # Logging
        self.log("val_loss", total_loss, sync_dist=True, batch_size=x.size(0))
        self.log("val_mse_loss", mse_loss, sync_dist=True, batch_size=x.size(0))
        self.log("val_kl_loss", kl_loss, sync_dist=True, batch_size=x.size(0))

        return {"loss": total_loss}

    def load_vae_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        vae_state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        self.model.load_state_dict(vae_state_dict)

    def configure_optimizers(self):
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