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

torch.set_warn_always(False)

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
    
    callbacks = [
        LatentSpacePCACallback(dataloader=dataloaders["val"], num_batches=4)
    ]

    # Create Lightning module
    pl_model = PseudoSpeakerVAE(**config["lightning"])

    # Create logger
    logger = TensorBoardLogger(**config["tensorboard"])

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
