import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from yaml import safe_load as yaml_safe_load

from . import create_classifier

dataset_metadata = yaml_safe_load(open("src/dataset/metadata.yaml", "r"))


class ImageClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # global configuration
        self.dataset_metadata = dataset_metadata[config["dataset"]]
        self.save_hyperparameters()

        # note: when the config num_classes is less than the total number of classes
        # in the dataset, we exclude some classes from train, eval and test data.
        # Here we pass the total number of classes to the model: we have more output
        # units than number of classes; then model needs to learn that some of the
        # classes never come up.
        model_config = config["model"]["config"]
        num_classes = self.dataset_metadata["num_classes"]
        in_channels = self.dataset_metadata["num_channels"]
        input_size = (
            self.dataset_metadata["height"]
            * self.dataset_metadata["width"]
            * self.dataset_metadata["num_channels"]
        )
        self.dataset_metadata["input_size"] = input_size
        self.model = create_classifier(
            config["model"]["architecture"],
            config=model_config,
            dataset_metadata=self.dataset_metadata,
        )

    def forward(self, x):
        return self.model(x)

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
        logits = self(x)
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
        logits = self(x)
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
