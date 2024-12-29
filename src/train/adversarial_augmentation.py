from collections import deque
from typing import Optional

import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F

from src.utils import get_layers


class AdversariallyAugmentedClassifier(pl.LightningModule):
    def __init__(self, config, classifier, autoencoder: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.classifier = classifier
        self.autoencoder = autoencoder
        if self.autoencoder is not None:
            self.autoencoder.eval()
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        self.layer_idx = config["layer_idx"]
        self.save_hyperparameters()
        self._split_classifier()

    def _split_classifier(self):
        """
        Splits the classifier into two consecutive chunks.
        Data augmentation will happen at the level of the output of the first chunk.
        """
        layers = get_layers(self.classifier)
        if self.layer_idx >= len(layers) or self.layer_idx < 0:
            raise ValueError(f"Layer index {self.layer_idx} is out of range")

        self.chunk1 = nn.Sequential(*layers[: self.layer_idx])
        self.chunk2 = nn.Sequential(*layers[self.layer_idx :])

    def forward(self, x, training=True):
        x = self.chunk1(x)
        if training:
            dx = self.compute_adversarial_perturbation(x.detach())
            x = x + dx
        x = self.chunk2(x)
        return x

    def compute_adversarial_perturbation(self, x):
        """
        Compute the logits of the classifier for x (output of chunk1), and perturb it in the
        direction that maximizes the logit of the second highest probability class.
        If an autoencoder is provided, first encode x, then compute the perturbation similarly
        in latent space. Return the resulting perturbation in the input space after decoding.
        """
        if self.autoencoder is None:
            x.requires_grad = True
            requires_grad = deque()
            for param in self.chunk2.parameters():
                requires_grad.append(param.requires_grad)
                param.requires_grad = False
            logits = self.chunk2(x)
            second_highest = logits.argsort(dim=1)[:, -2]  # TODO: check this
            obj = logits[:, second_highest].sum()
            obj.backward()
            for param in self.chunk2.parameters():
                param.requires_grad = requires_grad.popleft()
            return x.grad * self.config["epsilon"]
        else:
            z = self.autoencoder.encode(x)
            z.requires_grad = True
            requires_grad = deque()
            for param in self.chunk2.parameters():
                requires_grad.append(param.requires_grad)
                param.requires_grad = False
            logits = self.chunk2(self.autoencoder.decode(z))
            second_highest = logits.argsort(dim=1)[:, -2]
            obj = logits[:, second_highest].sum()
            obj.backward()
            for param in self.chunk2.parameters():
                param.requires_grad = requires_grad.popleft()
            return self.autoencoder.decode(z + z.grad * self.config["epsilon"]) - x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits, y, label_smoothing=self.config["training"]["label_smoothing"]
        )

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True)
        if batch_idx % self.config["logging"]["image_log_freq"] == 0:
            self._log_images(x, y, logits, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, training=False)
        loss = F.cross_entropy(
            logits, y, label_smoothing=self.config["training"]["label_smoothing"]
        )

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", acc, on_epoch=True)
        if batch_idx == 0:
            self._log_images(x, y, logits, "val")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, training=False)
        loss = F.cross_entropy(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)
        if batch_idx == 0:
            self._log_images(x, y, logits, "test")

        return loss

    def _log_images(self, x, y, logits, prefix, num_images=8):
        """
        Log images to wandb, with true and predicted labels.
        """
        if not self.config["logging"]["wandb_logging"]:
            return
        preds = logits.argmax(dim=1)

        if "class_names" in self.dataset_metadata:
            true_labels = [
                self.dataset_metadata["class_names"][y[i].item()]
                for i in range(min(num_images, len(x)))
            ]
            pred_labels = [
                self.dataset_metadata["class_names"][preds[i].item()]
                for i in range(min(num_images, len(x)))
            ]
        else:
            true_labels = [y[i].item() for i in range(min(num_images, len(x)))]
            pred_labels = [preds[i].item() for i in range(min(num_images, len(x)))]

        images = [
            wandb.Image(x[i], caption=f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
            for i in range(min(num_images, len(x)))
        ]

        self.logger.experiment.log({f"{prefix}-images": images})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        return optimizer
