from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.eval.interpolation_strategy import InterpolationStrategy
from src.eval.sampling_strategy import SamplingStrategy
from src.utils import load_from_hydra_logs, set_seed


class LatentExplorer:
    def __init__(
        self,
        autoencoder: nn.Module,
        dataloader: DataLoader,
        classifier: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the LatentExplorer.

        :param autoencoder: The trained autoencoder model
        :param dataloader: DataLoader for batch processing
        :param classifier: Optional classifier model for additional analysis
        :param device: Device to run computations on
        """
        self.autoencoder = autoencoder.to(device).eval()
        self.dataloader = dataloader
        self.classifier = classifier.to(device).eval() if classifier else None
        self.device = device

        self._cached_embeddings = None

    @classmethod
    def from_hydra_directory(cls, dir_path: str, model_class, device: str = "cuda"):
        """
        Load LatentExplorer from a Hydra output directory.

        :param dir_path: Path to the Hydra directory
        :param model_class: Class of the model to load
        :return: LatentExplorer instance
        """
        autoencoder, datamodule, config = load_from_hydra_logs(
            dir_path=dir_path, model_class=model_class
        )
        dataloader = DataLoader(
            datamodule.test_dataset,  # inherits transforms from config
            batch_size=config["data"]["batch_size"],  # type: ignore
            shuffle=True,
            num_workers=0,  # avoid issues with multiprocessing
        )
        autoencoder = autoencoder.to(device).eval()
        return cls(autoencoder, dataloader)

    def explore_around_image_in_latent_space(
        self,
        images: torch.Tensor,
        sampling_strategy: SamplingStrategy,
        num_points: int,
        config: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Embed a batch of images and sample points around their latent representations. Decode the samples.

        :param images: Batch of input images [B, C, H, W]
        :param sampling_strategy: Strategy to use for sampling
        :param num_points: Number of points to sample
        :param config: Configuration dictionary for the sampling strategy
        :param seed: Random seed for reproducibility
        :return: Dictionary containing:
            - 'originals': Original images, [B, C, H, W]
            - 'embeddings': Latent embeddings, [B, *embedding_dims]
            - 'samples': Sampled and decoded images, [num_points, B, C, H, W]
        """
        if seed is not None:
            set_seed(seed)
        images = images.to(self.device)
        batch_size = images.shape[0]
        with torch.no_grad():
            embeddings = self.autoencoder.encode(images)
        perturbed_embeddings = sampling_strategy.sample(
            embeddings, num_points, config, self.device
        )
        flat_embeddings = perturbed_embeddings.view(-1, *embeddings.shape[1:])
        with torch.no_grad():
            decoded_images = self.autoencoder.decode(flat_embeddings)
        decoded_images = decoded_images.view(
            num_points, batch_size, *decoded_images.shape[1:]
        )
        return {
            "originals": images,
            "embeddings": embeddings,
            "samples": decoded_images,
        }

    def explore_between_images_in_latent_space(
        self,
        start_images: torch.Tensor,
        end_images: torch.Tensor,
        interpolation_strategy: InterpolationStrategy,
        num_points: int,
        config: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Embed two batches of images and interpolate between their latent representations. Decode the interpolations.

        :param start_images: First batch of images [B, C, H, W]
        :param end_images: Second batch of images [B, C, H, W]
        :param interpolation_strategy: Strategy to use for interpolation
        :param num_points: Number of interpolation points (including start and end)
        :param config: Configuration dictionary for the interpolation strategy
        :param seed: Random seed for reproducibility
        :return: Dictionary containing:
            - 'start_images': Original starting images
            - 'end_images': Original ending images
            - 'start_embeddings': Starting point embeddings
            - 'end_embeddings': Ending point embeddings
            - 'interpolated': Interpolated and decoded images
        """
        if seed is not None:
            set_seed(seed)
        start_images = start_images.to(self.device)
        end_images = end_images.to(self.device)

        with torch.no_grad():
            start_embeddings = self.autoencoder.encode(start_images)
            end_embeddings = self.autoencoder.encode(end_images)
        interpolated_embeddings = interpolation_strategy.interpolate(
            start_embeddings, end_embeddings, num_points, config, self.device
        )

        batch_size = start_images.shape[0]
        flat_embeddings = interpolated_embeddings.view(-1, *start_embeddings.shape[1:])
        with torch.no_grad():
            decoded_images = self.autoencoder.decode(flat_embeddings)
        decoded_images = decoded_images.view(
            batch_size, num_points, *decoded_images.shape[1:]
        )

        return {
            "start_images": start_images,
            "end_images": end_images,
            "start_embeddings": start_embeddings,
            "end_embeddings": end_embeddings,
            "interpolated": decoded_images,
        }

    def embed_datapoints(
        self, num_samples: Optional[int] = None, seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute and cache embeddings for a specified number of samples from the dataloader.

        Args:
            num_samples: Number of samples to embed. If None, uses all samples.
            seed: Random seed for reproducibility.

        Returns:
            torch.Tensor: Cached embeddings of shape [N, *embedding_dims]
        """
        if seed is not None:
            torch.manual_seed(seed)

        embeddings_list = []
        samples_processed = 0

        with torch.no_grad():
            for batch in self.dataloader:
                # Handle different types of batch returns (tuple, dict, tensor)
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                elif isinstance(batch, dict):
                    images = batch["image"]
                else:
                    images = batch

                images = images.to(self.device)
                batch_embeddings = self.autoencoder.encode(images)
                embeddings_list.append(batch_embeddings.cpu())

                samples_processed += images.shape[0]
                if num_samples is not None and samples_processed >= num_samples:
                    # Trim the last batch if necessary
                    excess = samples_processed - num_samples
                    if excess > 0:
                        embeddings_list[-1] = embeddings_list[-1][:-excess]
                    break

        self._cached_embeddings = torch.cat(embeddings_list, dim=0)
        return self._cached_embeddings

    def get_latent_space_statistics(
        self,
        num_samples: Optional[int] = None,
        percentiles: list[float] = [1, 5, 25, 50, 75, 95, 99],
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute statistics about the latent space embeddings.

        Args:
            num_samples: Number of samples to use for statistics.
            percentiles: List of percentiles to compute.
            seed: Random seed for reproducibility.

        Returns:
            Dict containing various statistics about the latent space:
                - mean: Mean values per dimension
                - std: Standard deviation per dimension
                - percentiles: Specified percentiles per dimension
                - norms: Statistics about the embedding norms
                - correlations: Correlation matrix between dimensions
        """
        if self._cached_embeddings is None:
            self.embed_datapoints(num_samples=num_samples, seed=seed)
        embeddings = self._cached_embeddings.cpu().numpy()
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        percentile_values = np.percentile(embeddings, percentiles, axis=0)
        norms = np.linalg.norm(embeddings, axis=1)
        norm_stats = {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "norm_percentiles": {
                p: float(np.percentile(norms, p)) for p in percentiles
            },
        }
        correlations = np.corrcoef(embeddings.T)

        pca_results = None
        if embeddings.shape[1] > 2:
            pca = PCA()
            pca_transformed = pca.fit_transform(embeddings)
            pca_results = {
                "transformed": pca_transformed,
                "components": pca.components_,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "singular_values": pca.singular_values_,
            }

        dim_distributions = {
            "values": embeddings,  # Keep the raw values for detailed plotting
            "histograms": {
                i: np.histogram(embeddings[:, i], bins=50)
                for i in range(embeddings.shape[1])
            },
        }

        statistics = {
            "num_samples": len(embeddings),
            "dim": embeddings.shape[1],
            "mean": mean,
            "std": std,
            "percentiles": {
                p: values for p, values in zip(percentiles, percentile_values)
            },
            "norms": norm_stats,
            "correlations": correlations,
            "dimension_distributions": dim_distributions,
            "pca": pca_results,
        }
        return statistics

    def plot_latent_space_statistics(
        self,
        stats: Dict[str, Any],
        save_dir: Union[str, Path],
        dpi: int = 300,
        figsize_base: tuple[int, int] = (15, 8),
    ) -> None:
        """
        Generate and save visualizations of latent space statistics.

        Args:
            stats: Dictionary of statistics from get_latent_space_statistics
            save_dir: Directory to save plots
            dpi: DPI for saved figures
            figsize_base: Base figure size to use (will be adjusted for some plots)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Distribution of embedding norms
        plt.figure(figsize=figsize_base)
        norms_data = np.linalg.norm(stats["dimension_distributions"]["values"], axis=1)
        norms_data /= stats["dim"] ** 0.5
        sns.histplot(data=norms_data, bins=50, kde=True)
        plt.title("Norms of embeddings, normalized by sqrt(dim)")
        plt.xlabel("L2 Norm")
        plt.ylabel("Count")
        plt.savefig(
            save_dir / "normalized_norms_histogram.png", dpi=dpi, bbox_inches="tight"
        )
        plt.close()

        # 2. Correlation matrix heatmap
        plt.figure(figsize=(figsize_base[0], figsize_base[0]))  # Square figure
        sns.heatmap(
            stats["correlations"],
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=False,
            yticklabels=False,
        )
        plt.title("Dimension-wise Correlation Matrix")
        plt.savefig(
            save_dir / "embedding_coordinates_correlation_matrix.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        # 3. Distribution of values per dimension
        dim = stats["dim"]
        n_cols = min(4, dim)
        n_rows = min((dim + n_cols - 1) // n_cols, 2)
        num_plots = n_cols * n_rows
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize_base[0], figsize_base[1] * n_rows / 2),
            squeeze=False,
        )
        axes = axes.flatten()
        for i in range(num_plots):
            hist_counts, hist_bins = stats["dimension_distributions"]["histograms"][i]
            axes[i].stairs(hist_counts, hist_bins, fill=True)
            axes[i].set_title(f"Dimension {i}")
            axes[i].set_xlabel("Value")
        for i in range(dim, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            save_dir / "embedding_coordinates_histograms.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        # 4. Box plot of dimensions
        plt.figure(figsize=(max(figsize_base[0], num_plots / 2), figsize_base[1]))
        sns.boxplot(data=stats["dimension_distributions"]["values"])
        plt.title("Distribution of Values Across Dimensions")
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.xticks(rotation=90 if num_plots > 10 else 0)
        plt.savefig(
            save_dir / "embedding_coordinates_boxplots.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        # 5. Percentile plot
        plt.figure(figsize=figsize_base)
        percentiles = list(stats["percentiles"].keys())
        for dim_idx in range(min(10, num_plots)):
            values = [stats["percentiles"][p][dim_idx] for p in percentiles]
            plt.plot(percentiles, values, marker="o", label=f"Dim {dim_idx}")
        plt.title("Percentile Distribution by Dimension")
        plt.xlabel("Percentile")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.savefig(
            save_dir / "embedding_coordinates_percentiles.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        # 6. PCA explained variance
        if stats["pca"] is not None:
            plt.figure(figsize=figsize_base)
            plt.scatter(
                stats["pca"]["transformed"][:, 0],
                stats["pca"]["transformed"][:, 1],
                alpha=0.5,
                s=1,
            )
            plt.title("PCA Projection of Embeddings (2D)")
            plt.xlabel(
                f'PC1 ({stats["pca"]["explained_variance_ratio"][0]:.1%} variance)'
            )
            plt.ylabel(
                f'PC2 ({stats["pca"]["explained_variance_ratio"][1]:.1%} variance)'
            )
            plt.savefig(save_dir / "pca_projection.png", dpi=dpi, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=figsize_base)
            plt.plot(stats["pca"]["explained_variance_ratio"], "o-")
            plt.title("PCA Explained Variance Ratio")
            plt.xlabel("Principal Component")
            plt.ylabel("Explained Variance Ratio")
            plt.grid(True)
            plt.savefig(
                save_dir / "pca_explained_variance.png", dpi=dpi, bbox_inches="tight"
            )
            plt.close()

            plt.figure(figsize=figsize_base)
            plt.plot(np.cumsum(stats["pca"]["explained_variance_ratio"]), "o-")
            plt.title("Cumulative Explained Variance Ratio")
            plt.xlabel("Principal Component")
            plt.ylabel("Cumulative Explained Variance Ratio")
            plt.grid(True)
            plt.savefig(
                save_dir / "pca_cumulative_explained_variance.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close()
