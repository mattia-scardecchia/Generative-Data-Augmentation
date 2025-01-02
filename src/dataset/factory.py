def get_datamodule(config):
    """
    Factory function to get the appropriate datamodule.
    Use lazy imports to avoid circular imports.
    """
    from .cifar import CIFAR10DataModule, CIFAR100DataModule
    from .imagenet import TinyImageNetDataModule
    from .mnist import FashionMNISTDataModule, MNISTDataModule

    DATAMODULES = {
        "cifar10": CIFAR10DataModule,
        "cifar100": CIFAR100DataModule,
        "mnist": MNISTDataModule,
        "fashion_mnist": FashionMNISTDataModule,
        "tiny_imagenet": TinyImageNetDataModule,
    }

    dataset_name = config["dataset"].lower()
    if dataset_name not in DATAMODULES:
        raise ValueError(
            f"Dataset {dataset_name} not supported. Choose from: {list(DATAMODULES.keys())}"
        )
    datamodule = DATAMODULES[dataset_name](config)
    datamodule.set_metadata(dataset_name)
    return datamodule
