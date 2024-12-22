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
from src.models.classification.classifier import ImageClassifier
from src.utils import (
    get_class_names,
    load_from_hydra_logs,
)


@hydra.main(
    config_path="../configs/eval", config_name="input_gradients", version_base="1.3"
)
def main(cfg):
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    num_steps = cfg["num_steps"]
    optimizer_cls = getattr(torch.optim, cfg["optimizer_cls"])
    save_every_k = cfg["save_every_k"]
    hydra_path = cfg["hydra_path"]

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_dir = os.path.join(hydra_cfg["runtime"]["output_dir"], "input_grads")
    os.makedirs(save_dir, exist_ok=True)

    classifier, datamodule = load_from_hydra_logs(hydra_path, ImageClassifier)
    for param in classifier.parameters():
        param.requires_grad = False
    dataloader = DataLoader(
        datamodule.test_dataset,  # inherits transforms from config
        batch_size=2,
        shuffle=True,
        num_workers=0,  # avoid issues with multiprocessing
    )
    print("========== Model summary ==========")
    print(classifier)
    print(f"Len of Dataloader: {len(dataloader)}")
    x, y = next(iter(dataloader))
    class_names = get_class_names("fashion_mnist")
    print(f"class names: {class_names}")

    # Compute gradients of probabilities wrt input data
    grads = compute_all_probas_grads_wrt_data_and_plot(
        classifier, x.clone(), class_names=class_names
    )
    plt.savefig(os.path.join(save_dir, "gradients_wrt_inputs.png"))
    plt.close()

    # Optimize input data to maximize probabilities
    optimize_all_probas_wrt_data_and_plot(
        classifier,
        x.clone(),
        class_names,
        num_steps,
        optimizer_cls,
        lr=lr,
        weight_decay=weight_decay,
    )
    plt.savefig(os.path.join(save_dir, "optimized_inputs_for_probas.png"))
    plt.close()

    # Optimize random noise and zeros to maximize probabilities
    random_inputs = torch.randn_like(x)
    zeros = torch.zeros_like(x[0]).unsqueeze(0)
    data = torch.vstack([random_inputs, zeros])
    optimize_all_probas_wrt_data_and_plot(
        classifier,
        data,
        class_names,
        num_steps,
        optimizer_cls,
        lr=lr,
        weight_decay=weight_decay,
    )
    plt.savefig(os.path.join(save_dir, "optimal_inputs_for_probas.png"))
    plt.close()

    # Explore the optimization trajectory for a specific target class
    target_idx = 7
    traj, probas, grads = optimize_proba_wrt_data(
        classifier,
        x.clone(),
        target_idx,
        num_steps,
        optimizer_cls,
        save_every_k=save_every_k,
        lr=lr,
        weight_decay=weight_decay,
    )
    plot_optimization_metrics(probas, grads, target=class_names[target_idx])
    plt.savefig(os.path.join(save_dir, "proba_optimization_metrics.png"))
    plt.close()
    visualize_optimization_trajectory(probas, traj, target=class_names[target_idx])
    plt.savefig(os.path.join(save_dir, "proba_optimization_trajectory.png"))
    plt.close()


if __name__ == "__main__":
    main()
