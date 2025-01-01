import logging
import os

import hydra
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.input_gradients import (
    compute_proba_grads_wrt_data,
    optimize_proba_wrt_data,
    plot_grads_wrt_data,
    plot_optimal_images,
    plot_optimization_trajectory_fixed_target,
)
from src.models.autoencoding.autoencoder import Autoencoder
from src.models.classification.classifier import ImageClassifier
from src.utils import (
    get_class_names,
    load_from_hydra_logs,
    set_seed,
)


@hydra.main(
    config_path="../configs/eval", config_name="input_gradients", version_base="1.3"
)
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    base_save_dir = os.path.join(hydra_cfg["runtime"]["output_dir"], "input_grads")
    os.makedirs(base_save_dir, exist_ok=True)

    classifier, datamodule, config = load_from_hydra_logs(
        cfg["classifier_hydra_path"], ImageClassifier
    )
    classifier = classifier.to(cfg["device"])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    dataloader = DataLoader(
        datamodule.test_dataset,  # inherits transforms from config
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,  # avoid issues with multiprocessing
    )

    logging.info("========== Classifier summary ==========")
    logging.info(str(classifier))
    logging.info(f"Len of Dataloader: {len(dataloader)}")
    x, y = next(iter(dataloader))
    class_names = get_class_names(config["dataset"])
    logging.info(f"class names: {class_names}")

    targets = (
        list(range(len(class_names))) if cfg["targets"] is None else cfg["targets"]
    )
    optimizer_cls = getattr(torch.optim, cfg["optimizer"])

    if cfg["do_noae"]:
        set_seed(cfg["seed"])
        save_dir = os.path.join(base_save_dir, "pixels")
        os.makedirs(save_dir, exist_ok=True)

        # Compute gradients of probabilities wrt input data
        grads, finite_diffs = compute_proba_grads_wrt_data(
            classifier,
            x.clone(),
            targets,
            device=cfg["device"],
        )
        fig = plot_grads_wrt_data(
            x.clone(), grads, targets, class_names, figsize=cfg["figsize"]
        )
        fig.savefig(os.path.join(save_dir, "gradients_wrt_inputs.png"), dpi=cfg["dpi"])
        plt.close()

        # Optimize input data to maximize probabilities
        config = {
            "num_steps": cfg["num_steps"]["no_ae"],
            "optimizer_cls": optimizer_cls,
            "save_k_intermediate_imgs": cfg["save_k_intermediate_imgs"],
            "logit_transform": None,
            "optimizer_kwargs": {
                "lr": cfg["lr"]["no_ae"],
                "weight_decay": cfg["weight_decay"],
            },
        }
        out = optimize_proba_wrt_data(
            classifier,
            x.clone(),
            targets,
            config,
            cfg["device"],
        )
        fig = plot_optimal_images(out, class_names, figsize=cfg["figsize"])
        fig.savefig(
            os.path.join(save_dir, "optimized_inputs_for_probas.png"),
            dpi=cfg["dpi"],
        )
        plt.close()
        fig1, fig2 = plot_optimization_trajectory_fixed_target(
            out,
            cfg["fixed_target"],
            class_names,
            figsizes=(cfg["figsize"], cfg["figsize"]),
        )
        fig1.savefig(
            os.path.join(save_dir, "proba_optimization_metrics.png"),
            dpi=cfg["dpi"],
        )
        fig2.savefig(
            os.path.join(save_dir, "proba_optimization_trajectory.png"),
            dpi=cfg["dpi"],
        )
        plt.close()
        if cfg["save_tensors"]:
            torch.save(out, os.path.join(save_dir, "optimization_output.pt"))

        # optimize random noise and zeros to maximize probabilities
        random_inputs = torch.randn_like(x)
        zeros = torch.zeros_like(x[0]).unsqueeze(0)
        data = torch.vstack([random_inputs, zeros])
        out = optimize_proba_wrt_data(
            classifier,
            data,
            targets,
            config,
            cfg["device"],
        )
        fig = plot_optimal_images(out, class_names, figsize=cfg["figsize"])
        fig.savefig(
            os.path.join(save_dir, "optimized_noise_and_zeros_for_probas.png"),
            dpi=cfg["dpi"],
        )
        plt.close()
        fig1, fig2 = plot_optimization_trajectory_fixed_target(
            out,
            cfg["fixed_target"],
            class_names,
            figsizes=(cfg["figsize"], cfg["figsize"]),
        )
        fig1.savefig(
            os.path.join(save_dir, "proba_optimization_metrics_noise_and_zeros.png"),
            dpi=cfg["dpi"],
        )
        fig2.savefig(
            os.path.join(save_dir, "proba_optimization_trajectory_noise_and_zeros.png"),
            dpi=cfg["dpi"],
        )
        plt.close()

    if cfg["do_ae"]:
        if cfg["autoencoder_hydra_path"] is None:
            logging.error("Autoencoder hydra path is required when do_ae is True")
            return
        autoencoder, _, config = load_from_hydra_logs(
            cfg["autoencoder_hydra_path"], Autoencoder
        )
        autoencoder = autoencoder.to(cfg["device"])
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False
        logging.info("========== Autoencoder summary ==========")
        logging.info(str(autoencoder))

        set_seed(cfg["seed"])
        save_dir = os.path.join(base_save_dir, "manifold")
        os.makedirs(save_dir, exist_ok=True)

        # Compute gradients of probabilities wrt input data, on the AE manifold
        grads, finite_diffs = compute_proba_grads_wrt_data(
            classifier,
            x.clone(),
            targets,
            autoencoder=autoencoder,
            device=cfg["device"],
        )
        fig = plot_grads_wrt_data(
            x.clone(), finite_diffs, targets, class_names, figsize=cfg["figsize"]
        )
        fig.savefig(
            os.path.join(save_dir, "gradients_wrt_inputs.png"),
            dpi=cfg["dpi"],
        )
        plt.close()

        # Optimize input data to maximize probabilities, on the AE manifold
        config = {
            "num_steps": cfg["num_steps"]["ae"],
            "optimizer_cls": optimizer_cls,
            "save_k_intermediate_imgs": cfg["save_k_intermediate_imgs"],
            "logit_transform": None,
            "optimizer_kwargs": {
                "lr": cfg["lr"]["ae"],
                "weight_decay": cfg["weight_decay"],
            },
        }
        out = optimize_proba_wrt_data(
            classifier,
            x.clone(),
            targets,
            config,
            cfg["device"],
            autoencoder=autoencoder,
        )
        fig = plot_optimal_images(out, class_names, figsize=cfg["figsize"])
        fig.savefig(
            os.path.join(save_dir, "optimized_inputs_for_probas.png"),
            dpi=cfg["dpi"],
        )
        plt.close()
        fig1, fig2 = plot_optimization_trajectory_fixed_target(
            out,
            cfg["fixed_target"],
            class_names,
            figsizes=(cfg["figsize"], cfg["figsize"]),
        )
        fig1.savefig(
            os.path.join(save_dir, "proba_optimization_metrics.png"),
            dpi=cfg["dpi"],
        )
        fig2.savefig(
            os.path.join(save_dir, "proba_optimization_trajectory.png"),
            dpi=cfg["dpi"],
        )
        plt.close()
        if cfg["save_tensors"]:
            torch.save(out, os.path.join(save_dir, "optimization_output.pt"))

        # optimize random noise and zeros to maximize probabilities, on the AE manifold
        random_inputs = torch.randn_like(x)
        zeros = torch.zeros_like(x[0]).unsqueeze(0)
        data = torch.vstack([random_inputs, zeros])
        out = optimize_proba_wrt_data(
            classifier,
            data,
            targets,
            config,
            cfg["device"],
            autoencoder=autoencoder,
        )
        fig = plot_optimal_images(out, class_names, figsize=cfg["figsize"])
        fig.savefig(
            os.path.join(save_dir, "optimized_noise_and_zeros_for_probas.png"),
            dpi=cfg["dpi"],
        )
        plt.close()
        fig1, fig2 = plot_optimization_trajectory_fixed_target(
            out,
            cfg["fixed_target"],
            class_names,
            figsizes=(cfg["figsize"], cfg["figsize"]),
        )
        fig1.savefig(
            os.path.join(save_dir, "proba_optimization_metrics_noise_and_zeros.png"),
            dpi=cfg["dpi"],
        )
        fig2.savefig(
            os.path.join(save_dir, "proba_optimization_trajectory_noise_and_zeros.png"),
            dpi=cfg["dpi"],
        )
        plt.close()


if __name__ == "__main__":
    main()
