import hydra
from omegaconf import DictConfig
from train import train

from src.models.classification.classifier import ImageClassifier


@hydra.main(
    config_path="../configs/training", config_name="autoencoding", version_base="1.3"
)
def main(config: DictConfig):
    train(config, ImageClassifier)


if __name__ == "__main__":
    main()
