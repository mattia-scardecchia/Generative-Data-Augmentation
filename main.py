import datetime
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from pathlib import Path

from dataset import get_datamodule
from model.classifier import ImageClassifier
from utils import set_seed


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config("config.yaml")

    # Set seed for reproducibility
    set_seed(config["seed"])

    # Initialize data module
    datamodule = get_datamodule(config)

    # Initialize model
    model = ImageClassifier(config)

    # Initialize callbacks
    callbacks = []

    # Checkpoint directory
    checkpoint_dir = os.path.join(
        config["logging"]["checkpoint_root"],
        config["dataset"],
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
        every_n_epochs=config["logging"]["checkpoint_freq"],
    )
    callbacks.append(checkpoint_callback)

    # Initialize logger
    logger = None
    if config["logging"]["wandb_logging"]:
        logger = WandbLogger(
            project=config["logging"]["wandb_project"],
            entity=config["logging"]["wandb_entity"],
            config=config,
        )
        # Watch model gradients
        logger.watch(model, log="all", log_freq=config["logging"]["watch_freq"])

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu" if config["device"] == "cuda" else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=config["training"]["val_freq"],
        log_every_n_steps=config["logging"]["log_freq"],
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Test model
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
