import json
import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, early_stopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from ..dataset.factory import get_datamodule
from ..utils import set_seed


def train(
    config: DictConfig,
    model: nn.Module,
    datamodule: Optional[pl.LightningDataModule] = None,
):
    logging.getLogger("pytorch_lightning").propagate = True

    logging.info(f"Working directory: {os.getcwd()}")
    logging.info("========== Hydra config ==========")
    logging.info(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
    set_seed(config["seed"])
    logging.info("========== Model summary ==========")
    logging.info(str(model))
    if datamodule is None:
        datamodule = get_datamodule(config)

    callbacks = []
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    checkpoint_dir = os.path.join(
        hydra_cfg["runtime"]["output_dir"],
        config["logging"]["checkpoints"]["dirname"],
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        verbose=True,
        monitor=config["logging"]["checkpoints"]["monitor"],
        mode=config["logging"]["checkpoints"]["mode"],
        every_n_epochs=config["logging"]["checkpoints"]["checkpoint_freq"],
        save_top_k=config["logging"]["checkpoints"]["save_top_k"],
        save_last=config["logging"]["checkpoints"]["save_last"],
    )
    callbacks.append(checkpoint_callback)
    if config["early_stopping"]["do"]:
        early_stopping_callback = EarlyStopping(
            monitor=config["early_stopping"]["monitor"],
            mode=config["early_stopping"]["mode"],
            patience=config["early_stopping"]["patience"],
        )
        callbacks.append(early_stopping_callback)

    logger, run_id = None, None
    if config["logging"]["wandb_logging"]:
        logger = WandbLogger(
            project=config["logging"]["wandb_project"],
            entity=config["logging"]["wandb_entity"],
            config=OmegaConf.to_container(config, resolve=True),
        )
        logger.watch(model, log="all", log_freq=config["logging"]["watch_freq"])
        url = wandb.run.get_url()
        logging.info(f"Wandb Run URL: {url}")
        run_id = wandb.run.id
        logging.info(f"Wandb Run ID: {run_id}")

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu" if config["device"] == "cuda" else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=config["training"]["val_freq"],
        log_every_n_steps=config["logging"]["log_freq"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    if config["logging"]["wandb_logging"]:
        wandb.finish()

    return model, datamodule, trainer, run_id
