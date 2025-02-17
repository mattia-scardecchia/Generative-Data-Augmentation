import json
import logging
import os
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import wandb
import hydra
from src.models.autoencoding.variational_autoencoder import (
    KLAutoencoder,
)  # TODO: fix, make it flexibile
from src.models.classification.classifier import ImageClassifier
from src.utils import (
    set_seed,
    get_datamodule,
    iter_model_artifacts_from_runs,
    get_runs_from_tag,
)
from src.eval.ensemble_latent_explorer import EnsembleLatentExplorer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def perturb(autoencoder: KLAutoencoder, classifiers, data, config):
    if config["logging"]["wandb_logging"]:
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"]["wandb_entity"],
            config=OmegaConf.to_container(config, resolve=True),  # type: ignore
            tags=["vae-latentexplore"],
        )
        wandb.log(
            {
                "params/z_channels": autoencoder.z_channels,
                "params/kl_weight": autoencoder.kl_weight,
            }
        )
    logger.info("loading latent explorer")
    ensemble_latent_explorer = EnsembleLatentExplorer(autoencoder, classifiers)
    logger.info("starting gradient walk")
    perturbed_data = ensemble_latent_explorer(
        data=data,
        targets=config["perturb"]["target_classes"],
        lr=config["perturb"]["lr"],
    )
    logger.info("completed perturbation")
    wandb.finish()


@hydra.main(
    config_path="../configs/eval",
    config_name="ensemble_latent_space",
    version_base="1.3",
)
def main(config: DictConfig):
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("========== Hydra config ==========")
    logger.info(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
    set_seed(config["seed"])
    logger.info("========== Classifiers ===========")
    classifiers_runs = get_runs_from_tag(
        config["classifier_tag"],
        config["logging"]["wandb_project"],
        config["logging"]["wandb_entity"],
    )
    classifiers = [
        x[0] for x in iter_model_artifacts_from_runs(classifiers_runs, ImageClassifier)
    ]
    for classifier in classifiers:
        print(str(classifiers) + "\n\n")
    logger.info("========== Loading autoencoders ==========")
    autoencoders_runs = get_runs_from_tag(
        config["autoencoder_tag"],
        config["logging"]["wandb_project"],
        config["logging"]["wandb_entity"],
    )
    logger.info("========== Extracting data ==========")
    datamodule = get_datamodule(config)
    datamodule.setup(stage=None)
    data, labels = datamodule.extract_data(
        class_idx=config["data"]["starting_class"],
        num_images=config["data"]["sample_size"],
        split=config["data"]["split"],
        random=config["data"]["random"],
    )
    logging.info(f"Data class: {labels}")
    for autoencoder, _ in tqdm(
        iter_model_artifacts_from_runs(autoencoders_runs, KLAutoencoder)
    ):
        logger.info("========== Autoencoder summary ==========")
        logger.info(str(autoencoder))
        perturb(autoencoder, classifiers, data, config)


if __name__ == "__main__":
    main()
