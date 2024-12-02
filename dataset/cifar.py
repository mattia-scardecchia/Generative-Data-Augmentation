from torchvision import datasets, transforms
from .base import BaseDataModule


class CIFAR10DataModule(BaseDataModule):

    def prepare_data(self):
        datasets.CIFAR10(self.config["data_dir"], train=True, download=True)
        datasets.CIFAR10(self.config["data_dir"], train=False, download=True)

    def get_dataset(self, split: str, transform):
        return datasets.CIFAR10(
            root=self.config["data_dir"], train=(split == "train"), transform=transform
        )


class CIFAR100DataModule(BaseDataModule):

    def prepare_data(self):
        datasets.CIFAR100(self.config["data_dir"], train=True, download=True)
        datasets.CIFAR100(self.config["data_dir"], train=False, download=True)

    def get_dataset(self, split: str, transform):
        return datasets.CIFAR100(
            root=self.config["data_dir"], train=(split == "train"), transform=transform
        )
