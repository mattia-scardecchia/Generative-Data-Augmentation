from typing import Optional, Tuple, Literal, Any, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import yaml

from . import create_autoencoder

# structured output for encode, decode, etc...
# (x,), (z, mu, logvar) are examples
AutoencoderOutput = Tuple[torch.Tensor, *Tuple[Any, ...]]

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
                self.loss_fn = nn.BCEWithLogitsLoss()
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

    def forward(self, x: AutoencoderOutput) -> AutoencoderOutput:
        """
        Handles flattening of inputs if necessary. Output shape is the same as input shape.
        """
        img, *args = x
        input_shape = img.shape
        if self.flatten_inputs:
            img = img.flatten(1)
            x = (img, *args)
        out_img, *args = self.decode(self.encode(x))
        return (out_img.reshape(input_shape), *args)

    def training_step(self, batch, batch_idx):
        *x, y = batch
        x_hat = self(x)
        loss = self.compute_loss(x_hat, x, when="train")

        if (
            self.config["logging"]["image_log_freq"] is not None
            and batch_idx % self.config["logging"]["image_log_freq"] == 0
        ):
            self._log_images(x[0], y, x_hat[0], "train")

        return loss

    def validation_step(self, batch, batch_idx):
        *x, y = batch
        x_hat = self(x)
        loss = self.compute_loss(x_hat, x, when="val")

        if self.config["logging"]["image_log_freq"] is not None and batch_idx == 0:
            self._log_images(x[0], y, x_hat[0], "val")

        return loss

    def test_step(self, batch, batch_idx):
        *x, y = batch
        x_hat = self(x)
        loss = self.compute_loss(x_hat, x, when="test")

        if self.config["logging"]["image_log_freq"] is not None and batch_idx == 0:
            self._log_images(x[0], y, x_hat[0], "test")

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

    def encode(self, x: AutoencoderOutput) -> AutoencoderOutput:
        """
        Encode input images into latent space.
        update: instead of a single tensor, we allow to return a tuple
        of tensors for better compatibility, for example with KLAE
        we use huggingface-like logic

        args
        ----
        - x: 1-uple like (x,) where x is the image

        returns
        -------
        - (z,): 1-uple with latent representation
        """
        x_in, *_ = x
        return (self.encoder(x_in),)

    def decode(self, z: AutoencoderOutput) -> AutoencoderOutput:
        """
        the decode method should take the output of self.encode and produce an
        output as tuple.
        args
        ----
        - z: 1-uple like (z,), containing the latent
        returns
        -------
        - (x_hat,): 1-uple with reconstructed image
        """
        z_out, *_ = z
        return (self.decoder(z_out),)

    def compute_loss(
        self,
        x_hat: AutoencoderOutput,
        x: AutoencoderOutput,
        when: Literal["train", "val", "test"],
        log: bool = True,
    ) -> torch.Tensor:
        """isolating this step for better compatibility with variants. also logs"""
        loss = self.loss_fn(x_hat[0], x[0])
        if log:
            on_step = True if when == "train" else False
            self.log(f"{when}/loss", loss, on_step=on_step, on_epoch=True)
        return loss

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
