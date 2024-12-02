from abc import ABC, abstractmethod
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import torchvision as tv
import torch
import numpy as np


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config  # global configuration
        self.seed = self.config["seed"]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.metadata = None

    def set_metadata(self, metadata):
        for key, value in metadata.items():
            assert key not in self.__dict__, f"Metadata key {key} already exists"
            setattr(self, key, value)

    def setup_transforms(self):
        """
        Read data preprocessing parameters from config and create torchvision transforms
        for train and test datasets.
        """
        assert self.config["data"]["transforms"]["train"].keys() == self.config["data"]["transforms"]["test"].keys(), "Train and test transforms must have the same keys"
        transforms_names = self.config["data"]["transforms"]["train"].keys()
        for transform_name in transforms_names:
            transforms = []
            for phase in ["train", "test"]:
                params = self.config["data"]["transforms"][phase][transform_name]
                if params:
                    transform_class = getattr(tv.transforms, transform_name)
                    transforms.append(transform_class(**params))
            transforms.append(tv.transforms.ToTensor())
            if self.config["data"]["normalize"]:
                transforms.append(tv.transforms.Normalize(self.mean, self.std))
            setattr(self, f"{transform_name}_transform", tv.transforms.Compose(transforms))

    @abstractmethod
    def prepare_data(self):
        """Download dataset if needed"""
        pass

    def filter_dataset(self, dataset, num_classes, samples_per_class):
        """
        Select at random num_classes classes from the dataset, and for each of them
        select (at most) samples_per_class samples.
        """
        labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        sample_indices = []

        selected_classes = torch.randperm(
            len(dataset.classes),
            generator=torch.Generator().manual_seed(self.seed),
        )[:num_classes]

        for class_idx in selected_classes:
            class_indices = torch.where(labels == class_idx)[0]
            class_indices = class_indices[
                torch.randperm(
                    len(class_indices),
                    generator=torch.Generator().manual_seed(self.seed),
                )
            ]
            selected_indices = class_indices[:samples_per_class]
            sample_indices.extend(selected_indices.tolist())

        return Subset(dataset, sample_indices)

    def create_train_val_split(self, full_train_dataset):
        """Create train/val split from training dataset"""
        train_size = int(
            self.config["data"]["train_val_split"] * len(full_train_dataset)
        )
        val_size = len(full_train_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        return train_dataset, val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
        )
