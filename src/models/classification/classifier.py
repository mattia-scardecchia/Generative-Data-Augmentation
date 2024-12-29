from typing import Optional
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import wandb
from yaml import safe_load as yaml_safe_load

from . import create_classifier

dataset_metadata = yaml_safe_load(open("src/dataset/metadata.yaml", "r"))


class ImageClassifier(pl.LightningModule):
    def __init__(self, config, classifier: Optional[nn.Module] = None):
        """
        note: when the config num_classes is less than the total number of classes
        in the dataset, we exclude some classes from train, eval and test data.
        Here we pass the total number of classes to the model: we have more output
        units than number of classes; then model needs to learn that some of the
        classes never come up.
        """
        super().__init__()
        self.config = config  # global configuration
        self.dataset_metadata = dataset_metadata[config["dataset"]]
        self.save_hyperparameters()

        input_size = (
            self.dataset_metadata["height"]
            * self.dataset_metadata["width"]
            * self.dataset_metadata["num_channels"]
        )
        self.dataset_metadata["input_size"] = input_size
        
        if classifier is None:
            classifier = create_classifier(
                config["model"]["architecture"],
                config=config["model"]["config"],
                dataset_metadata=self.dataset_metadata,
            )
        self.classifier = classifier
        
        self.label_smoothing = config["training"]["label_smoothing"]
        self.lr = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.adam_eps = config["training"]["adam_eps"]
        self.image_log_freq = config["logging"]["image_log_freq"]
        self.wandb_logging = config["logging"]["wandb_logging"]

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits, y, label_smoothing=self.label_smoothing
        )

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True)
        if self.image_log_freq is not None and batch_idx % self.image_log_freq == 0:
            self._log_images(x, y, logits, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits, y, label_smoothing=self.label_smoothing
        )

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", acc, on_epoch=True)
        if self.image_log_freq is not None and batch_idx == 0:
            self._log_images(x, y, logits, "val")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)
        if self.image_log_freq is not None and batch_idx == 0:
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
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.adam_eps,
        )
        return optimizer
