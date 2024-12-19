from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import get_datamodule
from src.eval.interpolation_strategy import InterpolationStrategy
from src.eval.sampling_strategy import SamplingStrategy
from src.utils import set_seed


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

    @classmethod
    def from_hydra_directory(cls, dir_path: str, model_class, device: str = "cuda"):
        """
        Load LatentExplorer from a Hydra output directory.
        Assumes the following directory structure:
        .
        ├── .hydra
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── checkpoints
        │   └── last.ckpt
        └── train_ae.log

        :param dir_path: Path to the Hydra directory
        :param model_class: Class of the model to load
        :return: LatentExplorer instance
        """
        config = OmegaConf.load(f"{dir_path}/.hydra/config.yaml")
        autoencoder = model_class.load_from_checkpoint(
            f"{dir_path}/checkpoints/last.ckpt"
        )
        datamodule = get_datamodule(config)
        datamodule.setup()
        dataloader = DataLoader(
            datamodule.test_dataset,  # inherits transforms from config
            batch_size=config["data"]["batch_size"],
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
