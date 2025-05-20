import pytorch_lightning as pl
import torch.nn as nn
from ps_vae.model import VAEModel
from ps_vae.embedding_classifier.embedding_classifier import EmbeddingClassifier
from ps_vae.latent_classifier import LatentClassifier
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
            classifier_hparams = hparams["classifier"]
            self.classifier = LatentClassifier(**classifier_hparams)
            
            if isinstance(classifier_hparams['num_classes'], int): # Single label
                self.multilabel = False
                self.accuracy = Accuracy(task="multiclass", num_classes=hparams["classifier"]["num_classes"])
            
            else: # Multi-label
                self.multilabel = True
                self.accuracy = nn.ModuleDict({
                    label_name: Accuracy(task="multiclass", num_classes=label_classes) 
                    for label_name, label_classes in classifier_hparams['label_classes'].items()
                })
        else:
            self.classifier = None

        if 'consistency_classifier_ckpt' in hparams:
            self.consistency_classifier = EmbeddingClassifier.load_from_checkpoint(
                hparams['consistency_classifier_ckpt']
            )
            for param in self.consistency_classifier.parameters():
                param.requires_grad = False
            self.consistency_classifier.eval()
        else:
            self.consistency_classifier = None
        
        self.kl_loss_weight = hparams.get("kl_loss_weight", 1.0)
        self.classifier_loss_weight = hparams.get("classifier_loss_weight", 1.0)
        self.consitency_loss_weight = hparams.get("consistency_loss_weight", 1.0)
        self.use_cos_loss = hparams.get("use_cos_loss", False)

    def forward(self, x: torch.Tensor) -> tuple:
        x_hat, mu, log_sigma = self.model(x)
        return x_hat, mu, log_sigma
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.model.decode(z)
        return x_hat

    def training_step(self, batch: tuple, batch_idx: int) -> float:

        x, y = batch

        x_hat, mu, log_sigma = self(x)
        
        if self.classifier:
            
            y_hat = self.classifier(mu)

            if not self.multilabel:
                
                classifier_loss = nn.functional.cross_entropy(y_hat, y)
                classifier_acc = self.accuracy(y_hat, y)
                
                self.log("train_classifier_acc", classifier_acc, sync_dist=True)
                self.log("train_classifier_loss", classifier_loss, sync_dist=True)

            else:   
                classifier_loss = 0
                for label_name in y_hat:
                    
                    classifier_acc = self.accuracy[label_name](y_hat[label_name], y[label_name])
                    classifier_loss += nn.functional.cross_entropy(y_hat[label_name], y[label_name])
                    
                    self.log(f"train_classifier_acc_{label_name}", classifier_acc, sync_dist=True)
                    self.log(f"train_classifier_loss_{label_name}", classifier_loss, sync_dist=True)
                
                classifier_loss /= len(self.classifier.label_classes)
        
        else:
            classifier_loss = 0

        if self.consistency_classifier:
            y_hat_consistency = self.consistency_classifier(x_hat)
            consistency_loss = nn.functional.cross_entropy(y_hat_consistency, y)
            consistency_acc = self.accuracy(y_hat_consistency, y)
                
            self.log("train_consistency", consistency_acc, sync_dist=True)
            self.log("train_consistency_loss", consistency_loss, sync_dist=True)
        else:
            consistency_loss = 0

        if self.use_cos_loss:
            recon_loss = nn.functional.cosine_embedding_loss(x_hat, x, torch.ones(x.size(0)).to(x.device))
        else:
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")/10 # Roughly normalize the mse loss
        
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1)
        )

        total_loss = (
            recon_loss +
            self.kl_loss_weight * kl_loss + 
            self.classifier_loss_weight * classifier_loss +
            self.consitency_loss_weight * consistency_loss
        )
        
        # Logging
        self.log("train_loss", total_loss, sync_dist=True)
        self.log("train_recon_loss", recon_loss, sync_dist=True)
        self.log("train_kl_loss", kl_loss, sync_dist=True)

        return {"loss": total_loss}
    
    def validation_step(self, batch: tuple, batch_idx: int) -> float:

        x, y = batch
        
        x_hat, mu, log_sigma = self(x)
        
        if self.classifier:
            
            y_hat = self.classifier(mu)

            if not self.multilabel:
                
                classifier_loss = nn.functional.cross_entropy(y_hat, y)
                classifier_acc = self.accuracy(y_hat, y)
                
                self.log("val_classifier_acc", classifier_acc, sync_dist=True)
                self.log("val_classifier_loss", classifier_loss, sync_dist=True)

            else:   
                classifier_loss = 0
                for label_name in y_hat:
                    
                    classifier_acc = self.accuracy[label_name](y_hat[label_name], y[label_name])
                    classifier_loss += nn.functional.cross_entropy(y_hat[label_name], y[label_name])
                    
                    self.log(f"val_classifier_acc_{label_name}", classifier_acc, sync_dist=True)
                    self.log(f"val_classifier_loss_{label_name}", classifier_loss, sync_dist=True)
                
                classifier_loss /= len(self.classifier.label_classes)
        
        else:
            classifier_loss = 0

        if self.consistency_classifier:
            y_hat_consistency = self.consistency_classifier(x_hat)
            consistency_loss = nn.functional.cross_entropy(y_hat_consistency, y)
            consistency_acc = self.accuracy(y_hat_consistency, y)
                
            self.log("val_consistency", consistency_acc, sync_dist=True)
            self.log("val_consistency_loss", consistency_loss, sync_dist=True)
        else:
            consistency_loss = 0

        if self.use_cos_loss:
            recon_loss = nn.functional.cosine_embedding_loss(x_hat, x, torch.ones(x.size(0)).to(x.device))
        else:
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")/10 # Roughly normalize the mse loss
        
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1)
        )

        total_loss = (
            recon_loss +
            self.kl_loss_weight * kl_loss + 
            self.classifier_loss_weight * classifier_loss +
            self.consitency_loss_weight * consistency_loss
        )
        
        # Logging
        self.log("val_loss", total_loss, batch_size=x.size(0), sync_dist=True)
        self.log("val_recon_loss", recon_loss, batch_size=x.size(0), sync_dist=True)
        self.log("val_kl_loss", kl_loss, batch_size=x.size(0), sync_dist=True)

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