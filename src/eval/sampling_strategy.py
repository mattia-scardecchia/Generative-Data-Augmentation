from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from matplotlib import pyplot as plt

from src.utils import prepare_tensor_image_for_plot


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(
        self,
        reference_embedding: torch.Tensor,
        num_points: int,
        config: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Sample points around a reference embedding.

        Args:
            reference_embedding: The reference point in latent space [B, *embedding_dims]
            num_points: Number of points to sample per reference point
            config: Dictionary containing strategy-specific configuration
            device: Device to create tensors on

        Returns:
            Sampled points [B, num_points, *embedding_dims]
        """
        pass

    def plot_sampling_results(
        self,
        results: Dict[str, torch.Tensor],
        num_originals: int,
        num_samples: int,
        figsize: Tuple[int, int] = (15, 8),
    ) -> None:
        """
        Plot sampling results in a grid.

        :param results: Dictionary containing 'originals' and 'samples'.
        :param num_originals: Number of original images to display.
        :param num_samples: Number of samples to display per original.
        :param figsize: Figure size (width, height).
        """
        originals = results["originals"]  # [B, C, H, W]
        samples = results["samples"]  # [num_points, B, C, H, W]
        batch_size = originals.shape[0]
        max_samples = samples.shape[1]

        num_originals = min(num_originals, batch_size)
        num_samples = min(num_samples, max_samples)
        rows = num_originals
        cols = num_samples + 1  # +1 for original images
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes[None, :]
        for ax in axes.flat:
            ax.axis("off")

        for i in range(num_originals):
            axes[i, 0].imshow(prepare_tensor_image_for_plot(originals[i]))
            axes[i, 0].set_title("Original" if i == 0 else "")
            for j in range(num_samples):
                axes[i, j + 1].imshow(prepare_tensor_image_for_plot(samples[j, i]))
                if i == 0:
                    axes[i, j + 1].set_title(f"Sample {j+1}")
        plt.tight_layout()


class GaussianNoiseSampling(SamplingStrategy):
    def sample(
        self,
        reference_embedding: torch.Tensor,
        num_points: int,
        config: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Sample points around a reference embedding by adding Gaussian noise.
        :param config: dictionary containing the following keys: stddev.
        """
        stddev = config.get("stddev", 0.1)
        batch_size = reference_embedding.size(0)
        embedding_shape = reference_embedding.shape[1:]

        noise = (
            torch.randn((num_points, batch_size, *embedding_shape), device=device)
            * stddev
        )
        perturbations = reference_embedding.unsqueeze(0) + noise
        return perturbations


class SphericalSampling(SamplingStrategy):
    def sample(
        self,
        reference_embedding: torch.Tensor,
        num_points: int,
        config: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        radius = config.get("radius", 1.0)
        batch_size = reference_embedding.shape[0]
        embedding_dim = reference_embedding.shape[1]

        # Generate random directions
        directions = torch.randn(batch_size, num_points, embedding_dim, device=device)
        # Normalize directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        # Scale by radius
        points = reference_embedding.unsqueeze(1) + directions * radius

        return points
