import logging
from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision as tv
from torch.utils.data import DataLoader, Dataset, Subset
from yaml import safe_load as yaml_safe_load

from src.dataset.factory import get_datamodule


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config  # global configuration
        self.seed = self.config["seed"]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.metadata = None

        # set by set_metadata
        self.mean = None
        self.std = None
        self.num_classes = None
        self.num_channels = None
        self.height = None
        self.width = None
        self.class_names = None

        self.persistent_workers = self.config["data"].get("persistent_workers", False)

    def set_metadata(self, dataset_name: str, metadata=None):
        """
        Read metadata from file and set attributes. Expects that attributes are
        initialized to None in the constructor.
        """
        all_metadata = yaml_safe_load(open("src/dataset/metadata.yaml", "r"))
        assert dataset_name in all_metadata, f"Metadata for {dataset_name} not found"
        for key, value in all_metadata[dataset_name].items():
            assert key in self.__dict__, f"Trying to set unknown metadata key {key}"
            curr = getattr(self, key)
            assert curr is None, f"Metadata key {key} already set to {curr}"
            setattr(self, key, value)

    def setup_transforms(self):
        """
        Read data preprocessing parameters from config and create torchvision transforms
        for train and test datasets.
        """
        assert (
            self.config["data"]["transforms"]["train"].keys()
            == self.config["data"]["transforms"]["test"].keys()
        ), "Train and test transforms must have the same keys"
        transforms_names = self.config["data"]["transforms"]["train"].keys()
        for phase in ["train", "test"]:
            transforms = []
            for transform_name in transforms_names:
                params = self.config["data"]["transforms"][phase][transform_name]
                if params:
                    transform_class = getattr(tv.transforms, transform_name)
                    transforms.append(transform_class(**params))
            transforms.append(tv.transforms.ToTensor())
            if self.config["data"]["transforms"]["normalize"]:
                transforms.append(tv.transforms.Normalize(self.mean, self.std))
            setattr(self, f"{phase}_transform", tv.transforms.Compose(transforms))

    @abstractmethod
    def prepare_data(self):
        """Download dataset if needed"""
        pass

    @abstractmethod
    def get_dataset(self, split: str, transform) -> Dataset:
        """Return dataset for a given split"""
        pass

    def setup(self, stage=None):
        self.setup_transforms()

        if stage == "fit" or stage is None:
            full_train_dataset = self.get_dataset("train", self.train_transform)
            if (
                self.config["num_classes"] != -1
                or self.config["data"]["samples_per_class"] != -1
            ):
                full_train_dataset = self.filter_dataset(
                    full_train_dataset,
                    num_classes=self.config["num_classes"],
                    samples_per_class=self.config["data"]["samples_per_class"],
                )
            self.train_dataset, self.val_dataset = self.create_train_val_split(
                full_train_dataset
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.get_dataset("test", self.test_transform)
            if (
                self.config["num_classes"] != -1
                or self.config["data"]["samples_per_class"] != -1
            ):
                self.test_dataset = self.filter_dataset(
                    self.test_dataset,
                    num_classes=self.config["num_classes"],
                    samples_per_class=self.config["data"]["samples_per_class"],
                )

    def filter_dataset(self, dataset, num_classes, samples_per_class):
        """
        Select at random num_classes classes from the dataset, and for each of them
        select (at most) samples_per_class samples.
        """
        labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        sample_indices = []

        if num_classes == -1:
            selected_classes = torch.unique(labels)
        else:
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
            selected_indices = (
                class_indices[:samples_per_class]
                if samples_per_class > 0
                else class_indices
            )
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
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            persistent_workers=self.persistent_workers,
        )
        logging.info(f"Train dataloader: {len(dl)} batches of size {dl.batch_size}")
        return dl

    def val_dataloader(self, shuffle=False):
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["data"]["num_workers"],
            persistent_workers=self.persistent_workers,
        )
        logging.info(f"Val dataloader: {len(dl)} batches of size {dl.batch_size}")
        return dl

    def test_dataloader(self, shuffle=False):
        dl = DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["data"]["num_workers"],
            persistent_workers=self.persistent_workers,
        )
        logging.info(f"Test dataloader: {len(dl)} batches of size {dl.batch_size}")
        return dl

    @staticmethod
    def get_default_dataset(
        dataset: str, num_classes: Optional[int] = 10, samples_per_class: int = -1
    ):
        config = {
            "dataset": dataset,
            "num_classes": num_classes,
            "data_dir": "./data/datasets",
            "seed": 42,
            "data": {
                "samples_per_class": samples_per_class,
                "batch_size": 64,
                "num_workers": 0,
                "train_val_split": 0.8,
                "transforms": {
                    "normalize": True,
                    "train": {
                        "RandomHorizontalFlip": {"p": 0.0},
                        "RandomRotation": {"degrees": 0},
                        "RandomCrop": None,
                    },
                    "test": {
                        "RandomHorizontalFlip": None,
                        "RandomRotation": None,
                        "RandomCrop": None,
                    },
                },
            },
        }
        datamodule = get_datamodule(config)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule
