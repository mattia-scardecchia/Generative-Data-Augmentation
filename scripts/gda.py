import hydra
from omegaconf import DictConfig

import wandb
from src.dataset.hidden_representations import HiddenRepresentationModule
from src.models.autoencoding.autoencoder import Autoencoder
from src.models.classification.classifier import ImageClassifier
from src.train.adversarial_augmentation import AdversariallyAugmentedClassifier
from src.train.train import train

# TODO: expose relevant parameters to the config (e.g. epsilon)
# TODO: create finetuninng_config in the config file
# TODO: add some info-level logging to know what's going on


@hydra.main(config_path="../configs/gda", config_name="gda", version_base="1.3")
def main(config: DictConfig):
    """
    - train a classifier
    - train several autoencoders to reconstruct the hidden representations of the classifier
      at various layers
    - save the trained models
    - have a baseline by finetuning the classifier on the original data
    - for each layer, do generative data augmentation with and without the autoencoder
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
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    classifier_chkpt_dir = f"{output_dir}/classifier_checkpoints"

    for layer_idx in config["layers"]:
        classifier = ImageClassifier.load_from_checkpoint(
            f"{classifier_chkpt_dir}/last.ckpt"
        )  # play safe
        hidden_datamodule = HiddenRepresentationModule(
            classifier,
            layer_idx,
            datamodule,
            config["autoencoder_config"]["data"]["batch_size"],
            config["autoencoder_config"]["data"]["num_workers"],
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

    # baseline
    classifier = ImageClassifier.load_from_checkpoint(
        f"{classifier_chkpt_dir}/last.ckpt"
    )
    train(config["finetuning_config"], classifier, datamodule)

    # generative data augmentation
    for layer_idx in config["layers"]:
        # without autoencoder
        classifier = ImageClassifier.load_from_checkpoint(
            f"{classifier_chkpt_dir}/last.ckpt"
        )
        classifier_with_gda = AdversariallyAugmentedClassifier(
            config={"epsilon": 0.1, "layer_idx": layer_idx}, classifier=classifier
        )
        train(config["finetuning_config"], classifier_with_gda, datamodule)

        # with autoencoder
        classifier = ImageClassifier.load_from_checkpoint(
            f"{classifier_chkpt_dir}/last.ckpt"
        )
        autoencoder = Autoencoder.load_from_checkpoint(
            f"{output_dir}/autoencoder_{layer_idx}_checkpoints/last.ckpt"
        )
        classifier_with_gda = AdversariallyAugmentedClassifier(
            config={"epsilon": 0.1, "layer_idx": layer_idx},
            classifier=classifier,
            autoencoder=autoencoder,
        )
        train(config["finetuning_config"], classifier_with_gda, datamodule)


if __name__ == "__main__":
    main()
