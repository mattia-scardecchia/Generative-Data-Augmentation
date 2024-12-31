import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from src.utils import prepare_tensor_image_for_plot


def inference_with_autoencoder(autoencoder, dataloader, device="cuda"):
    """
    Do batched inference with a PyTorch Lightning autoencoder on a provided dataloader's data.
    Assumes the autoencoder has an 'encode' method.

    Args:
        autoencoder: PyTorch Lightning Module (the autoencoder model)
        dataloader: PyTorch DataLoader
        device: str, one of ['cpu', 'cuda']

    Returns:
        dict: dictionary containing the following keys:
            - 'mse_losses': array of MSE losses for each sample
            - 'inputs': original input data
            - 'latent_vectors': encoded representations
            - 'reconstructions': reconstructed outputs
    """
    autoencoder.to(device).eval()
    all_inputs = []
    all_latent = []
    all_reconstructions = []
    all_mse_losses = []
    mse_criterion = torch.nn.MSELoss(reduction="none")

    with torch.no_grad():
        for x in tqdm(
            dataloader, desc=f"Doing inference on {len(dataloader)} batches on {device}"
        ):
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)

            latent = autoencoder.encode(x)
            reconstruction = autoencoder.decode(latent)
            mse_losses = mse_criterion(reconstruction, x)
            mse_losses = mse_losses.mean(dim=tuple(range(1, len(mse_losses.shape))))

            all_inputs.append(x.cpu().numpy())
            all_latent.append(latent.cpu().numpy())
            all_reconstructions.append(reconstruction.cpu().numpy())
            all_mse_losses.append(mse_losses.cpu().numpy())

    return {
        "inputs": np.concatenate(all_inputs),
        "latent_vectors": np.concatenate(all_latent),
        "reconstructions": np.concatenate(all_reconstructions),
        "mse_losses": np.concatenate(all_mse_losses),
    }


def collect_high_error_reconstructions(
    autoencoder, dataloader, device, threshold, num_samples=9
):
    """
    Collect examples where autoencoder reconstruction error exceeds a threshold.

    Args:
        autoencoder: PyTorch autoencoder model in eval mode
        dataloader: PyTorch DataLoader containing the validation/test data
        device: device to run computations on
        threshold: MSE threshold above which to collect samples
        num_samples: number of high-error samples to collect (default=9)

    Returns:
        tuple: (original_inputs, reconstructions, latent_codes, mse_values)
        - original_inputs: list of tensors containing original input images
        - reconstructions: list of tensors containing reconstructed images
        - latent_codes: list of tensors containing latent representations
        - mse_values: list of MSE values for collected samples
    """
    autoencoder.to(device).eval()
    original_inputs = []
    reconstructions = []
    latent_codes = []
    mse_values = []
    mse_criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            latents = autoencoder.encode(images)
            recons = autoencoder.decode(latents)
            mse = mse_criterion(recons, images).view(images.size(0), -1).mean(dim=1)
            high_error_idx = torch.where(mse > threshold)[0]

            for idx in high_error_idx:
                if len(original_inputs) >= num_samples:
                    break
                original_inputs.append(images[idx].cpu())
                reconstructions.append(recons[idx].cpu())
                latent_codes.append(latents[idx].cpu())
                mse_values.append(mse[idx].item())

            if len(original_inputs) >= num_samples:
                break

    return original_inputs, reconstructions, latent_codes, mse_values


def plot_reconstruction_pairs(
    inputs, reconstructions, num_pairs=None, figsize=(15, 8), mse_losses=None
):
    """
    Plot pairs of original images and their reconstructions from an autoencoder.

    Args:
        inputs: numpy array of original images
        reconstructions: numpy array of reconstructed images
        num_pairs: int, number of pairs to plot (default: 3)
        figsize: tuple, figure size in inches (default: (15, 8))
        mse_losses: numpy array of MSE losses for each sample (default: None)
    """
    assert len(inputs) == len(
        reconstructions
    ), "Inputs and reconstructions must have the same length"
    if mse_losses is not None:
        assert len(inputs) == len(
            mse_losses
        ), "Inputs and MSE losses must have the same length"
    if num_pairs is None:
        num_pairs = len(inputs)
    num_pairs = min(num_pairs, len(inputs))
    fig, axes = plt.subplots(num_pairs, 3, figsize=figsize, squeeze=False)

    for idx in range(num_pairs):
        axes[idx, 0].imshow(prepare_tensor_image_for_plot(inputs[idx]))
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title("original")
        axes[idx, 1].imshow(prepare_tensor_image_for_plot(reconstructions[idx]))
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title("reconstruction")
        axes[idx, 2].imshow(
            prepare_tensor_image_for_plot(reconstructions[idx] - inputs[idx])
        )
        axes[idx, 2].axis("off")
        title = "diff"
        if mse_losses is not None:
            title += f" (mse {mse_losses[idx]:.4f})"
        axes[idx, 2].set_title(title)

    plt.tight_layout()
    return fig
