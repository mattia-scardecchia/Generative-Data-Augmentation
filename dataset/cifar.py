from torchvision import datasets, transforms
from .base import BaseDataModule


class CIFAR10DataModule(BaseDataModule):

    def setup_transforms(self):
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def prepare_data(self):
        datasets.CIFAR10(self.config["data_dir"], train=True, download=True)
        datasets.CIFAR10(self.config["data_dir"], train=False, download=True)

    def setup(self, stage=None):
        self.setup_transforms()

        if stage == "fit" or stage is None:
            full_train_dataset = datasets.CIFAR10(
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
            test_dataset = datasets.CIFAR10(
                root=self.config["data_dir"], train=False, transform=self.test_transform
            )
            self.test_dataset = self.filter_dataset(
                test_dataset,
                num_classes=self.config["num_classes"],
                samples_per_class=self.config["data"]["samples_per_class"],
            )


class CIFAR100DataModule(BaseDataModule):

    def setup_transforms(self):
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def prepare_data(self):
        datasets.CIFAR100(self.config["data_dir"], train=True, download=True)
        datasets.CIFAR100(self.config["data_dir"], train=False, download=True)

    def setup(self, stage=None):
        self.setup_transforms()

        if stage == "fit" or stage is None:
            full_train_dataset = datasets.CIFAR100(
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
            test_dataset = datasets.CIFAR100(
                root=self.config["data_dir"], train=False, transform=self.test_transform
            )
            self.test_dataset = self.filter_dataset(
                test_dataset,
                num_classes=self.config["num_classes"],
                samples_per_class=self.config["data"]["samples_per_class"],
            )
