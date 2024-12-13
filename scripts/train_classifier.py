import datetime
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset import get_datamodule
from src.models.classification.classifier import ImageClassifier
from src.utils import set_seed


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@hydra.main(config_path="../configs", config_name="classification")
def main(config: DictConfig):
    for key, value in config.items():
        print(key, value)
    set_seed(config["seed"])
    datamodule = get_datamodule(config)
    model = ImageClassifier(config)
    callbacks = []
    checkpoint_dir = os.path.join(
        config["logging"]["checkpoint_root"],
        config["dataset"],
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
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
    logger = None
    if config["logging"]["wandb_logging"]:
        logger = WandbLogger(
            project=config["logging"]["wandb_project"],
            entity=config["logging"]["wandb_entity"],
            config=OmegaConf.to_container(config, resolve=True),
        )
        logger.watch(model, log="all", log_freq=config["logging"]["watch_freq"])
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu" if config["device"] == "cuda" else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=config["training"]["val_freq"],
        log_every_n_steps=config["logging"]["log_freq"],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
