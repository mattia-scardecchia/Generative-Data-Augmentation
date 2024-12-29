import logging
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.utils import get_layers


class HiddenRepresentationDataset(Dataset):
    def __init__(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        """
        Dataset class for storing hidden representations and their labels.

        Args:
            hidden_states: Tensor of hidden representations
            labels: Tensor of corresponding labels
        """
        self.hidden_states = hidden_states
        self.labels = labels

    def __len__(self) -> int:
        return len(self.hidden_states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hidden_states[idx], self.labels[idx]


class HiddenRepresentationModule(LightningDataModule):
    def __init__(
        self,
        classifier: nn.Module,
        layer_idx: int,
        datamodule: LightningDataModule,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        DataModule for creating and managing hidden representation datasets.

        Args:
            classifier: The classifier model to extract representations from
            layer_idx: Index of the layer to extract representations from
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
        """
        super().__init__()
        self.classifier = classifier
        self.layer_idx = layer_idx
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.datamodule = datamodule
        self.partial_model, self.input_shape = self._create_partial_model()

        logging.info(f"Extracting hidden representations from layer {layer_idx}")
        logging.info(f"Partial model:\n {self.partial_model}")
        logging.info(f"Input shape: {self.input_shape}")

    def _create_partial_model(self) -> Tuple[nn.Module, Tuple[int, ...]]:
        """Create a model that outputs the hidden representations at the specified layer."""
        layers = get_layers(self.classifier)
        if self.layer_idx >= len(layers) or self.layer_idx < 0:
            raise ValueError(f"Layer index {self.layer_idx} is out of range")

        layers = list(layers[: self.layer_idx])
        partial_model = nn.Sequential(*layers)
        with torch.inference_mode():
            data = self.datamodule.train_dataset[0][0]
            hidden = partial_model(data.unsqueeze(0))
            input_shape = hidden.shape[1:]

        return partial_model, input_shape

    def _extract_hidden_states(self, dataset: Dataset) -> Dataset:
        """Extract hidden representations from the dataset using the partial model."""
        hidden_states = []
        labels = []
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self.partial_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, batch_labels = batch
                batch_hidden = self.partial_model(inputs)
                hidden_states.append(batch_hidden)
                labels.append(batch_labels)

        hidden_states = torch.cat(hidden_states)
        labels = torch.cat(labels)
        return HiddenRepresentationDataset(hidden_states, labels)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._extract_hidden_states(
                self.datamodule.train_dataset
            )
            self.val_dataset = self._extract_hidden_states(self.datamodule.val_dataset)
        if stage == "test" or stage is None:
            self.test_dataset = self._extract_hidden_states(
                self.datamodule.test_dataset
            )

    def train_dataloader(self) -> Optional[DataLoader]:
        """Get the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Get the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
