"""Pytorch Lightning Training Script for a simple embedding classifier."""

import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
from ps_vae.utils import load_yaml_config
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from ps_vae.data.cv import get_cv_dataloaders
from ps_vae.data.vctk import get_vctk_dataloaders
from ps_vae.embedding_classifier.embedding_classifier import EmbeddingClassifier

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os

torch.set_warn_always(False)
torch.set_float32_matmul_precision('medium')

# Parse command line arguments
parser = argparse.ArgumentParser(description="PyTorch Lightning Training Script")
parser.add_argument(
    "config", type=str, help="Path to the YAML configuration file"
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
    dataset_name = config["dataset"].pop("name", "cv")
    if dataset_name == "cv":
        dataloaders = get_cv_dataloaders(
            dataset_kwargs=config["dataset"], **config["dataloader"]
        )
    elif dataset_name == "vctk":
        dataloaders = get_vctk_dataloaders(
            dataset_kwargs=config["dataset"], **config["dataloader"]
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # Create Lightning module
    pl_model = EmbeddingClassifier(**config["lightning"])

    # Create logger
    logger = TensorBoardLogger(**config["tensorboard"])

    # Add ModelCheckpoint callback to save best model based on val loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="best-checkpoint",
        save_top_k=1,
        mode="min"
    )

    # Make trainer
    trainer = Trainer(
        logger=logger,
        callbacks=checkpoint_callback,
        **config["trainer"]
    )

    trainer.fit(
        pl_model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=config["ckpt_path"],
    )

    
