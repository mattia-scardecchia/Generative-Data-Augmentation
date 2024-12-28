import hydra
from omegaconf import DictConfig

from src.models.classification.classifier import ImageClassifier
from src.train.train import train


@hydra.main(
    config_path="../configs/training", config_name="classification", version_base="1.3"
)
def main(config: DictConfig):
    model = ImageClassifier(config)
    train(config, model)


if __name__ == "__main__":
    main()
