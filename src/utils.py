import math
import random
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from src.dataset.factory import get_datamodule


def get_layers(model):
    layers = []
    for layer in model.modules():
        if len(list(layer.children())) == 0:  # If module has no children, it's a layer
            layers.append(layer)
    return layers


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_tensor_image_for_plot(img: torch.Tensor) -> np.ndarray:
    """
    Prepare an image tensor for plotting by converting to numpy array and normalizing.

    :param img: Input image tensor [C, H, W]
    :return: Prepared image as numpy array [H, W, C]
    """
    numpy_img = img.cpu().detach().numpy()

    # Ensure the image is in range [0, 1]
    if numpy_img.max() > 1.0 or numpy_img.min() < 0.0:
        normalized_img = (numpy_img - numpy_img.min()) / (
            numpy_img.max() - numpy_img.min()
        )
    else:
        normalized_img = numpy_img

    # Rearrange dimensions from [C, H, W] to [H, W, C]
    transposed_img = np.transpose(normalized_img, (1, 2, 0))

    # Convert grayscale to RGB
    if transposed_img.shape[2] == 1:
        final_img = np.repeat(transposed_img, 3, axis=2)
    else:
        final_img = transposed_img

    return final_img


def load_from_hydra_logs(dir_path: str, model_class):
    """
    Loads model and datamodule from hydra logs.
    Assumes the following directory structure:
        .
        ├── .hydra
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── checkpoints
        │   └── last.ckpt
        └── train_ae.log
    """
    config = OmegaConf.load(f"{dir_path}/.hydra/config.yaml")
    model = model_class.load_from_checkpoint(f"{dir_path}/checkpoints/last.ckpt")
    datamodule = get_datamodule(config)
    datamodule.setup()
    return model, datamodule, config


def get_class_names(dataset: str, metadata_path: str = "src/dataset/metadata.yaml"):
    """
    Retrieve class names from src/dataset/metadata.yaml and return them.
    """
    metadata = OmegaConf.load(metadata_path)
    if dataset not in metadata:
        raise ValueError(f"Dataset {dataset} not found in metadata.")
    return metadata[dataset].class_names


def get_mean_and_std(dataset: str, metadata_path: str = "src/dataset/metadata.yaml"):
    """
    Retrieve mean and standard deviation from src/dataset/metadata.yaml and return them.
    """
    metadata = OmegaConf.load(metadata_path)
    if dataset not in metadata:
        raise ValueError(f"Dataset {dataset} not found in metadata.")
    return metadata[dataset].mean, metadata[dataset].std


def plot_image_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[list[str]] = None,
    figsize=(15, 15),
):
    """
    Plot a grid of images with their labels from batched tensors.

    Args:
        images: Tensor of shape (B, C, H, W) where B is batch size
        labels: Tensor of shape (B,) containing class indices
        class_names: optional list of class names to use instead of indices
    """
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Batch sizes of images and labels must match")

    n_images = images.shape[0]
    grid_size = math.ceil(math.sqrt(n_images))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx in range(len(axes)):
        if idx < n_images:
            label = labels[idx].item()
            axes[idx].imshow(prepare_tensor_image_for_plot(images[idx]))
            label_text = class_names[label] if class_names is not None else str(label)
            axes[idx].set_title(f"{label_text}")
        axes[idx].axis("off")

    plt.tight_layout()
    return fig
