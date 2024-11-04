from .cifar import CIFAR10DataModule, CIFAR100DataModule
from .mnist import MNISTDataModule, FashionMNISTDataModule
from yaml import safe_load as yaml_safe_load

DATAMODULES = {
    "cifar10": CIFAR10DataModule,
    "cifar100": CIFAR100DataModule,
    "mnist": MNISTDataModule,
    "fashion_mnist": FashionMNISTDataModule,
}

metadata = yaml_safe_load(open("dataset/metadata.yaml", "r"))


def get_datamodule(config):
    """Factory function to get the appropriate datamodule"""
    dataset_name = config["dataset"].lower()
    if dataset_name not in DATAMODULES:
        raise ValueError(
            f"Dataset {dataset_name} not supported. Choose from: {list(DATAMODULES.keys())}"
        )

    datamodule = DATAMODULES[dataset_name](config)
    datamodule.set_metadata(metadata[dataset_name])
    return datamodule
