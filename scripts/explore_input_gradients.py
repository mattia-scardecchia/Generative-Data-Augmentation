import os

import hydra
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.input_gradients import (
    compute_all_probas_grads_wrt_data_and_plot,
    optimize_all_probas_wrt_data_and_plot,
    optimize_proba_wrt_data,
    plot_optimization_metrics,
    visualize_optimization_trajectory,
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
    optimizer_cls = getattr(torch.optim, cfg["optimizer"])
    save_every_k = cfg["save_every_k"]
    if save_every_k == "auto":
        save_every_k = cfg["num_steps"] // 10

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_dir = os.path.join(hydra_cfg["runtime"]["output_dir"], "input_grads")
    os.makedirs(save_dir, exist_ok=True)

    classifier, datamodule, config = load_from_hydra_logs(
        cfg["classifier_hydra_path"], ImageClassifier
    )
    classifier = classifier.to(cfg["device"])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    set_seed(cfg["seed"])
    dataloader = DataLoader(
        datamodule.test_dataset,  # inherits transforms from config
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,  # avoid issues with multiprocessing
    )
    print("========== Classifier summary ==========")
    print(classifier)
    print(f"Len of Dataloader: {len(dataloader)}")
    x, y = next(iter(dataloader))
    class_names = get_class_names(config["dataset"])
    print(f"class names: {class_names}")

    # Compute gradients of probabilities wrt input data
    grads = compute_all_probas_grads_wrt_data_and_plot(
        classifier, x.clone(), class_names=class_names
    )
    plt.savefig(os.path.join(save_dir, "gradients_wrt_inputs.png"), dpi=cfg["dpi"])
    plt.close()

    # Optimize input data to maximize probabilities
    optimize_all_probas_wrt_data_and_plot(
        classifier,
        x.clone(),
        class_names,
        None,
        cfg["num_steps"],
        optimizer_cls,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    plt.savefig(
        os.path.join(save_dir, "optimized_inputs_for_probas.png"), dpi=cfg["dpi"]
    )
    plt.close()

    # Optimize random noise and zeros to maximize probabilities
    random_inputs = torch.randn_like(x)
    zeros = torch.zeros_like(x[0]).unsqueeze(0)
    data = torch.vstack([random_inputs, zeros])
    optimize_all_probas_wrt_data_and_plot(
        classifier,
        data,
        class_names,
        None,
        cfg["num_steps"],
        optimizer_cls,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    plt.savefig(os.path.join(save_dir, "optimal_inputs_for_probas.png"), dpi=cfg["dpi"])
    plt.close()

    # Explore the optimization trajectory for a specific target class
    target_idx = cfg["target_class"]
    traj, probas, grads = optimize_proba_wrt_data(
        classifier,
        x.clone(),
        target_idx,
        cfg["num_steps"],
        optimizer_cls,
        save_every_k=save_every_k,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    plot_optimization_metrics(probas, grads, target=class_names[target_idx])
    plt.savefig(
        os.path.join(save_dir, "proba_optimization_metrics.png"), dpi=cfg["dpi"]
    )
    plt.close()
    visualize_optimization_trajectory(probas, traj, target=class_names[target_idx])
    plt.savefig(
        os.path.join(save_dir, "proba_optimization_trajectory.png"), dpi=cfg["dpi"]
    )
    plt.close()

    if cfg["autoencoder_hydra_path"] is None:
        return

    autoencoder, _, config = load_from_hydra_logs(
        cfg["autoencoder_hydra_path"], Autoencoder
    )
    autoencoder = autoencoder.to(cfg["device"])
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    set_seed(cfg["seed"])
    print("========== Autoencoder summary ==========")
    print(autoencoder)

    # Optimize input data to maximize probabilities, on the manifold learned by an autoencoder
    optimize_all_probas_wrt_data_and_plot(
        classifier,
        x.clone(),
        class_names,
        autoencoder=autoencoder,
        num_steps=cfg["num_steps"],
        optimizer_cls=optimizer_cls,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    plt.savefig(
        os.path.join(save_dir, "optimized_inputs_for_probas_on_manifold.png"),
        dpi=cfg["dpi"],
    )
    plt.close()


if __name__ == "__main__":
    main()
