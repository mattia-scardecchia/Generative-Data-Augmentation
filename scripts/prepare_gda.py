import hydra
from omegaconf import DictConfig

import wandb
from src.dataset.hidden_representations import HiddenRepresentationModule
from src.models.autoencoding.autoencoder import Autoencoder
from src.models.classification.classifier import ImageClassifier
from src.train.train import train

# TODO: expose relevant parameters to the config
# TODO: add some info logging to know what's going on


@hydra.main(config_path="../configs/gda", config_name="prepare_gda", version_base="1.3")
def main(config: DictConfig):
    """
    - train a classifier
    - train several autoencoders to reconstruct the hidden representations of the classifier
      at various layers
    - save the trained models
    """
    classifier = ImageClassifier(config["classifier_config"])
    classifier, datamodule, _, classifier_run_id = train(
        config["classifier_config"], classifier
    )
    wandb_logging = (
        config["autoencoder_config"]["logging"]["wandb_logging"]
        and config["classifier_config"]["logging"]["wandb_logging"]
    )
    if wandb_logging:
        api = wandb.Api()
        parent_run = api.run(
            f"{config['classifier_config']['logging']['wandb_entity']}/{config['classifier_config']['logging']['wandb_project']}/{classifier_run_id}"
        )  # TODO: check that entity works correctly

    for layer_idx in config["layers"]:
        hidden_datamodule = HiddenRepresentationModule(
            classifier, layer_idx, datamodule
        )
        config["autoencoder_config"]["logging"]["checkpoints"]["dirname"] = (
            f"autoencoder_{layer_idx}_checkpoints"
        )
        autoencoder = Autoencoder(
            config["autoencoder_config"], input_shape=hidden_datamodule.input_shape
        )
        autoencoder, _, _, autoencoder_run_id = train(
            config["autoencoder_config"], autoencoder, hidden_datamodule
        )
        if wandb_logging:
            derived_run = api.run(
                f"{config['autoencoder_config']['logging']['wandb_entity']}/{config['autoencoder_config']['logging']['wandb_project']}/{autoencoder_run_id}"
            )
            derived_run.config["parent_run_id"] = parent_run.id
            derived_run.update()


if __name__ == "__main__":
    main()
