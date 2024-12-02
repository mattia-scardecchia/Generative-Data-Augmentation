from torchvision import datasets, transforms
from .base import BaseDataModule


class MNISTDataModule(BaseDataModule):

    def prepare_data(self):
        datasets.MNIST(self.config["data_dir"], train=True, download=True)
        datasets.MNIST(self.config["data_dir"], train=False, download=True)

    def get_dataset(self, split: str, transform):
        return datasets.MNIST(
            root=self.config["data_dir"], train=(split == "train"), transform=transform
        )


class FashionMNISTDataModule(BaseDataModule):

    def prepare_data(self):
        datasets.FashionMNIST(self.config["data_dir"], train=True, download=True)
        datasets.FashionMNIST(self.config["data_dir"], train=False, download=True)

    def get_dataset(self, split: str, transform):
        return datasets.FashionMNIST(
            root=self.config["data_dir"], train=(split == "train"), transform=transform
        )
