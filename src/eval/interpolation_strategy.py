from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from matplotlib import pyplot as plt

from src.utils import prepare_tensor_image_for_plot


class InterpolationStrategy(ABC):
    @abstractmethod
    def interpolate(
        self,
        start_embedding: torch.Tensor,
        end_embedding: torch.Tensor,
        num_points: int,
        config: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Interpolate between two embeddings.

        Args:
            start_embedding: Starting point in latent space [B, *embedding_dims]
            end_embedding: Ending point in latent space [B, *embedding_dims]
            num_points: Number of interpolation points (including start and end)
            config: Dictionary containing strategy-specific configuration
            device: Device to create tensors on

        Returns:
            Interpolated points [B, num_points, *embedding_dims]
        """
        pass

    def plot_interpolation_results(
        self,
        results: Dict[str, torch.Tensor],
        num_pairs: int = 5,
        figsize: Tuple[int, int] = (15, 8),
    ) -> None:
        """
        Plot interpolation results in rows.

        :param results: Dictionary containing interpolation results
        :param num_pairs: Number of examples to plot
        :param figsize: Figure size (width, height)
        """
        start_images = results["start_images"]
        interpolated = results["interpolated"]
        batch_size = start_images.shape[0]
        num_steps = interpolated.shape[1]

        num_pairs = min(num_pairs, batch_size)
        rows = num_pairs
        cols = num_steps
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if rows == 1:
            axes = axes[None, :]
        for ax in axes.flat:
            ax.axis("off")

        for i in range(num_pairs):
            for j in range(num_steps):
                axes[i, j].imshow(prepare_tensor_image_for_plot(interpolated[i, j]))
                if i == 0:
                    if j == 0:
                        axes[i, j].set_title("Start")
                    elif j == num_steps - 1:
                        axes[i, j].set_title("End")
                    else:
                        axes[i, j].set_title(f"{j/(num_steps-1):.2f}")

        plt.tight_layout()


class LinearInterpolation(InterpolationStrategy):
    def interpolate(
        self,
        start_embedding: torch.Tensor,
        end_embedding: torch.Tensor,
        num_points: int,
        config: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        t = torch.linspace(0, 1, num_points, device=device)
        embedding_shape = start_embedding.shape[1:]
        ones = [1 for _ in embedding_shape]

        # Reshape for broadcasting
        start_embedding = start_embedding.unsqueeze(1)  # [B, 1, *embedding_shape]
        end_embedding = end_embedding.unsqueeze(1)  # [B, 1, *embedding_shape]
        t = t.view(1, num_points, *ones)

        return (1 - t) * start_embedding + t * end_embedding


class SphericalInterpolation(InterpolationStrategy):
    def interpolate(
        self,
        start_embedding: torch.Tensor,
        end_embedding: torch.Tensor,
        num_points: int,
        config: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        # Normalize embeddings to lie on unit sphere
        start_norm = torch.norm(start_embedding, dim=-1, keepdim=True)
        end_norm = torch.norm(end_embedding, dim=-1, keepdim=True)

        start_normalized = start_embedding / start_norm
        end_normalized = end_embedding / end_norm

        # Compute cosine of the angle between the vectors
        cos_omega = torch.sum(start_normalized * end_normalized, dim=-1, keepdim=True)
        omega = torch.acos(torch.clamp(cos_omega, -1 + 1e-7, 1 - 1e-7))

        # Generate interpolation coefficients
        t = torch.linspace(0, 1, num_points, device=device).view(1, -1, 1)

        # Compute spherical interpolation
        sin_omega = torch.sin(omega)
        start_coeff = torch.sin((1 - t) * omega) / sin_omega
        end_coeff = torch.sin(t * omega) / sin_omega

        # Interpolate the norms linearly
        norm_t = (1 - t) * start_norm + t * end_norm

        # Combine everything
        interpolated = start_coeff * start_embedding.unsqueeze(
            1
        ) + end_coeff * end_embedding.unsqueeze(1)

        # Scale by interpolated norm
        return interpolated * norm_t
