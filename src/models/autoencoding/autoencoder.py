from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import yaml

from . import create_autoencoder

dataset_metadata = yaml.safe_load(open("src/dataset/metadata.yaml", "r"))


class Autoencoder(pl.LightningModule):
    def __init__(self, config, input_shape: Optional[tuple[int]] = None):
        super().__init__()
        self.config = config
        self.dataset_metadata = dataset_metadata[config["dataset"]]
        if input_shape is not None:
            self._update_input_shape_in_metadata(input_shape)
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

        self.adam_eps = config["training"].get("adam_eps", 1e-8)

    def _update_input_shape_in_metadata(self, input_shape):
        """
        Useful to train on hidden representations (shape is different from original images).
        """
        # TODO: check order C,H,W is correct/general
        self.dataset_metadata["height"] = input_shape[-2]
        self.dataset_metadata["width"] = input_shape[-1]
        self.dataset_metadata["num_channels"] = input_shape[-3]

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
        if (
            self.config["logging"]["image_log_freq"] is not None
            and batch_idx % self.config["logging"]["image_log_freq"] == 0
        ):
            self._log_images(x, y, x_hat, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)

        self.log("val/loss", loss, on_epoch=True)
        if self.config["logging"]["image_log_freq"] is not None and batch_idx == 0:
            self._log_images(x, y, x_hat, "val")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)

        self.log("test/loss", loss, on_epoch=True)
        if self.config["logging"]["image_log_freq"] is not None and batch_idx == 0:
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
                f"Original vs. Reconstructed, {true_labels[i]}"
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
            eps=self.adam_eps,
        )
        return optimizer

    def encode(self, x):
        """
        Encode input images into latent space.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent vectors into images.
        """
        return self.decoder(z)

    @staticmethod
    def default_config():
        return {
            "model": {
                "architecture": "conv",
                "config": {
                    "block_type": "convnext",
                    "hidden_dims": [128, 128],
                    "latent_dim": 32,
                    "downsample": [2, 2],
                    "block_kwargs": {},
                },
            },
            "dataset": "cifar10",
            "training": {
                "epochs": 50,
                "learning_rate": 0.001,
                "gradient_clip_val": 1.0,
                "weight_decay": 0.0001,
                "val_freq": 1.0,  # fraction of epoch (float) or steps (int)
                "loss": "mse",
            },
            "logging": {
                "wandb_logging": False,
            },
        }
