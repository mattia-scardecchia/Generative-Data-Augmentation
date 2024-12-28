import hydra
from omegaconf import DictConfig

from src.models.autoencoding.autoencoder import Autoencoder
from src.train.train import train


@hydra.main(
    config_path="../configs/training", config_name="autoencoding", version_base="1.3"
)
def main(config: DictConfig):
    model = Autoencoder(config)
    train(config, model)


if __name__ == "__main__":
    main()
