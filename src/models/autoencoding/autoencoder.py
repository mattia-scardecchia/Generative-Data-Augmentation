import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml

import wandb

from . import create_autoencoder

dataset_metadata = yaml.safe_load(open("src/dataset/metadata.yaml", "r"))


class Autoencoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_metadata = dataset_metadata[config["dataset"]]
        self.save_hyperparameters()
        self.flatten_inputs = self.config["model"]["architecture"] in ["mlp"]

        model_config = config["model"]["config"]
        in_channels = self.dataset_metadata["num_channels"]
        input_size = (
            self.dataset_metadata["height"]
            * self.dataset_metadata["width"]
            * self.dataset_metadata["num_channels"]
        )

        self.encoder: nn.Module
        self.decoder: nn.Module
        self.encoder, self.decoder = create_autoencoder(
            architecture=config["model"]["architecture"],
            config=model_config,
            in_channels=in_channels,
            input_size=input_size,
        )

        match config["training"]["loss"].lower():
            case "mse":
                self.loss_fn = nn.MSELoss()
            case "bce":
                self.loss_fn = nn.BCELoss()
            case "l1":
                self.loss_fn = nn.L1Loss()
            case "smoothl1":
                self.loss_fn = nn.SmoothL1Loss()
            case _:
                raise ValueError(f"Unknown loss function: {config['training']['loss']}")

    def forward(self, x):
        """
        Handles flattening of inputs if necessary. Output shape is the same as input shape.
        """
        input_shape = x.shape
        if self.flatten_inputs:
            x = x.flatten(1)
        out = self.decoder(self.encoder(x))
        return out.reshape(input_shape)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        if (batch_idx + 1) % self.config["logging"]["image_log_freq"] == 0:
            self._log_images(x, y, x_hat, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)

        self.log("val/loss", loss, on_epoch=True)
        if (batch_idx + 1) % self.config["logging"]["image_log_freq"] == 0:
            self._log_images(x, y, x_hat, "val")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)

        self.log("test/loss", loss, on_epoch=True)
        if (batch_idx + 1) % self.config["logging"]["image_log_freq"] == 0:
            self._log_images(x, y, x_hat, "test")

        return loss

    def _log_images(self, x, y, x_hat, prefix, num_images=8):
        """
        Log original and reconstructed images to wandb.
        Include true labels if available.
        """
        if not self.config["logging"]["wandb_logging"]:
            return

        if "class_names" in self.dataset_metadata:
            true_labels = [self.dataset_metadata["class_names"][label] for label in y]
        else:
            true_labels = None

        images = []
        for i in range(num_images):
            concatenated_image = torch.cat([x[i], x_hat[i]], dim=2)
            caption = (
                f"Original vs. Reconstructed, {y[i]}"
                if true_labels
                else "Original vs. Reconstructed"
            )
            images.append(wandb.Image(concatenated_image, caption=caption))

        wandb.log({f"{prefix}/images": images})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        return optimizer
