import copy
import hydra
from omegaconf import DictConfig

from src.train import adversarial_augmentation
import wandb
from src.dataset.hidden_representations import HiddenRepresentationModule
from src.models.autoencoding.autoencoder import Autoencoder
from src.models.classification.classifier import ImageClassifier
from src.train.adversarial_augmentation import AdversariallyAugmentedClassifier
from src.train.train import train


# TODO: rethink config system for gda, there's too many repetitions and stuff that is ignored
# TODO: maybe make a train function specific for gda?
# TODO: make two scripts, one for preparing the models and another for the gda (to tune hyperparameters)


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

    # ==================== classifier training ==================== #

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
        )
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    classifier_chkpt_dir = f"{output_dir}/classifier_checkpoints"

    # ==================== autoencoders training ==================== #

    for layer_idx in config["layers"]:
        classifier = ImageClassifier.load_from_checkpoint(
            f"{classifier_chkpt_dir}/last.ckpt"
        )
        hidden_datamodule = HiddenRepresentationModule(
            classifier,
            layer_idx,
            datamodule,
            config["hidden_representations_dataset"],
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

    # ==================== classifier finetuning with gda ==================== #

    # baseline
    classifier = ImageClassifier.load_from_checkpoint(
        f"{classifier_chkpt_dir}/last.ckpt"
    )
    train(config["finetuning_config"], classifier, datamodule)

    # gda
    for layer_idx in config["layers"]:
        finetuning_config = copy.deepcopy(config["finetuning_config"])
        finetuning_config["layer_idx"] = layer_idx

        # without autoencoder
        classifier = ImageClassifier.load_from_checkpoint(
            f"{classifier_chkpt_dir}/last.ckpt"
        )
        classifier_with_gda = AdversariallyAugmentedClassifier(
            config=finetuning_config, classifier=classifier
        )
        train(finetuning_config, classifier_with_gda, datamodule)

        # with autoencoder
        classifier = ImageClassifier.load_from_checkpoint(
            f"{classifier_chkpt_dir}/last.ckpt"
        )
        autoencoder = Autoencoder.load_from_checkpoint(
            f"{output_dir}/autoencoder_{layer_idx}_checkpoints/last.ckpt"
        )
        classifier_with_gda = AdversariallyAugmentedClassifier(
            config=finetuning_config,
            classifier=classifier,
            autoencoder=autoencoder,
        )
        train(finetuning_config, classifier_with_gda, datamodule)


if __name__ == "__main__":
    main()
