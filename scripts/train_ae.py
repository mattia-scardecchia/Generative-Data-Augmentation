import yaml
import hydra
from omegaconf import DictConfig
from src.models.autoencoding.autoencoder import Autoencoder


@hydra.main(config_path="../configs", config_name="autoencoding")
def main(cfg: DictConfig):
    for key, value in cfg.items():
        print(f"{key}: {value}")
    model = Autoencoder(cfg)
    print(model)


if __name__ == "__main__":
    main()