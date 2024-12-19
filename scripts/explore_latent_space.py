import os

from matplotlib import pyplot as plt

from src.eval.interpolation_strategy import LinearInterpolation
from src.eval.latent_explorer import LatentExplorer
from src.eval.sampling_strategy import GaussianNoiseSampling
from src.models.autoencoding.autoencoder import Autoencoder

hydra_path = "/Users/mat/Desktop/Files/Code/Playground/cfg:dataset=cifar10,model.config.latent_dim=32,training.epochs=50/seed=12"
explorer = LatentExplorer.from_hydra_directory(hydra_path, Autoencoder, device="mps")
print(explorer.autoencoder)
print(len(explorer.dataloader))
images = next(iter(explorer.dataloader))[0]


strategy = GaussianNoiseSampling()
out = explorer.explore_around_image_in_latent_space(
    images,
    strategy,
    3,
    config={"stddev": 0.5},
    seed=12,
)
strategy.plot_sampling_results(out, 5, 3)
path = "data/figures/gaussian_noise_sampling.png"
os.makedirs(os.path.dirname(path), exist_ok=True)
plt.savefig(path)
plt.close()


im1 = next(iter(explorer.dataloader))[0][:4]
im2 = next(iter(explorer.dataloader))[0][4:8]
strategy = LinearInterpolation()
out = explorer.explore_between_images_in_latent_space(im1, im2, strategy, 10, {})
strategy.plot_interpolation_results(out, 4)
path = "data/figures/linear_interpolation.png"
os.makedirs(os.path.dirname(path), exist_ok=True)
plt.savefig(path)
plt.close()
