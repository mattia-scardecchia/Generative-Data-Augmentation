import json
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.dataset import get_datamodule
from src.utils import set_seed


def train(config: DictConfig, model_class):
    print(f"Working directory: {os.getcwd()}")
    print("Hydra config:")
    print(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
    set_seed(config["seed"])
    datamodule = get_datamodule(config)
    model = model_class(config)

    callbacks = []
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    checkpoint_dir = os.path.join(
        hydra_cfg["runtime"]["output_dir"],
        "checkpoints",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{val_loss:.2f}",
        save_last=True,
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
