"""Pytorch Lightning Training Script for PS-VAE"""

import argparse
from utils import load_yaml_config, LatentSpacePCACallback
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from ps_vae.data import get_dataloaders
from ps_vae.lightning import PseudoSpeakerVAE

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_warn_always(False)
torch.set_float32_matmul_precision('medium')

# Parse command line arguments
parser = argparse.ArgumentParser(description="PyTorch Lightning Training Script")
parser.add_argument(
    "--config", type=str, required=True, help="Path to the YAML configuration file"
)
args = parser.parse_args()

CONFIG_PATH = args.config

if __name__ == "__main__":

    # Load config, and perform general setup
    config = load_yaml_config(CONFIG_PATH)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    # Setup dataloaders
    dataloaders = get_dataloaders(
        dataset_kwargs=config["dataset"], **config["dataloader"]
    )
    
    
    # Create logger
    logger = TensorBoardLogger(**config["tensorboard"])
    
    # Setup callbacks
    callbacks = []

    if config["pca_batches"]:
        callbacks.append(
            LatentSpacePCACallback(dataloader=dataloaders["val"], num_batches=config["pca_batches"])
        )

    # Add checkpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
        mode="min"
    )
    callbacks.append(checkpoint_callback)

    # Create Lightning module
    pl_model = PseudoSpeakerVAE(**config["lightning"])

    # Make trainer
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
        **config["trainer"]
    )

    trainer.fit(
        pl_model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=config["ckpt_path"],
    )
