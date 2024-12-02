from torchvision import datasets, transforms
from .base import BaseDataModule


class MNISTDataModule(BaseDataModule):

    def prepare_data(self):
        datasets.MNIST(self.config["data_dir"], train=True, download=True)
        datasets.MNIST(self.config["data_dir"], train=False, download=True)

    def setup(self, stage=None):
        self.setup_transforms()

        if stage == "fit" or stage is None:
            full_train_dataset = datasets.MNIST(
                root=self.config["data_dir"], train=True, transform=self.train_transform
            )
            full_train_dataset = self.filter_dataset(
                full_train_dataset,
                num_classes=self.config["num_classes"],
                samples_per_class=self.config["data"]["samples_per_class"],
            )
            self.train_dataset, self.val_dataset = self.create_train_val_split(
                full_train_dataset
            )

        if stage == "test" or stage is None:
            test_dataset = datasets.MNIST(
                root=self.config["data_dir"], train=False, transform=self.test_transform
            )
            self.test_dataset = self.filter_dataset(
                test_dataset,
                num_classes=self.config["num_classes"],
                samples_per_class=self.config["data"]["samples_per_class"],
            )


class FashionMNISTDataModule(BaseDataModule):

    def prepare_data(self):
        datasets.FashionMNIST(self.config["data_dir"], train=True, download=True)
        datasets.FashionMNIST(self.config["data_dir"], train=False, download=True)

    def setup(self, stage=None):
        self.setup_transforms()

        if stage == "fit" or stage is None:
            full_train_dataset = datasets.FashionMNIST(
                root=self.config["data_dir"], train=True, transform=self.train_transform
            )
            full_train_dataset = self.filter_dataset(
                full_train_dataset,
                num_classes=self.config["num_classes"],
                samples_per_class=self.config["data"]["samples_per_class"],
            )
            self.train_dataset, self.val_dataset = self.create_train_val_split(
                full_train_dataset
            )

        if stage == "test" or stage is None:
            test_dataset = datasets.FashionMNIST(
                root=self.config["data_dir"], train=False, transform=self.test_transform
            )
            self.test_dataset = self.filter_dataset(
                test_dataset,
                num_classes=self.config["num_classes"],
                samples_per_class=self.config["data"]["samples_per_class"],
            )
