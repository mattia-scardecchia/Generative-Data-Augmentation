import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from src.eval.interpolation_strategy import LinearInterpolation
from src.eval.latent_explorer import LatentExplorer
from src.eval.sampling_strategy import GaussianNoiseSampling
from src.models.autoencoding.autoencoder import Autoencoder

hydra_path = "autoencoders/convae-fashion_mnist/2024-12-27/20-02-13"
explorer = LatentExplorer.from_hydra_directory(
    hydra_path,
    Autoencoder,
    device="mps",
)
logging.info("============= Autoencoder Summary =============")
logging.info(explorer.autoencoder)
logging.info(f"Save directory: {explorer.save_dir}")
images = next(iter(explorer.dataloader))[0]


batch_size = 2
num_exploration_samples = 3
num_interpolation_points = 10
num_samples_statistics = 300
figsize = (30, 15)


for stddev in np.linspace(0.0, 1.0, 11)[1:]:
    stddev = round(stddev, 2)
    strategy = GaussianNoiseSampling()
    out = explorer.explore_around_image_in_latent_space(
        images,
        strategy,
        3,
        config={"stddev": stddev},
        seed=12,
    )
    strategy.plot_sampling_results(
        out, batch_size, num_exploration_samples, figsize=figsize
    )
    path = explorer.save_dir / f"gaussian_noise_sampling_{stddev}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


im1 = next(iter(explorer.dataloader))[0][:batch_size]
im2 = next(iter(explorer.dataloader))[0][batch_size : 2 * batch_size]
strategy = LinearInterpolation()
out = explorer.explore_between_images_in_latent_space(
    im1, im2, strategy, num_interpolation_points, {}
)
strategy.plot_interpolation_results(out, batch_size)
path = explorer.save_dir / "linear_interpolation.png"
os.makedirs(os.path.dirname(path), exist_ok=True)
plt.savefig(path)
plt.close()


statistics = explorer.get_latent_space_statistics(num_samples=num_samples_statistics)
explorer.plot_latent_space_statistics(statistics)
