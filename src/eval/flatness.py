import copy
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import prepare_tensor_image_for_plot


@torch.no_grad()
def perturb_weights(
    classifier: nn.Module,
    stddev: float,
):
    """
    Perturb the weights of a neural network classifier with multiplicative Gaussian noise.
    Ignore BatchNorm layers.
    """
    for name, param in classifier.named_parameters():
        if "BatchNorm" in name:
            continue
        noise = torch.randn_like(param) * stddev
        param.mul_(1 + noise)


@torch.no_grad()
def compute_local_energy(
    classifier: nn.Module,
    dataloader: DataLoader,
    num_samples: Optional[int] = None,
    loss_fn: Optional[nn.Module] = None,
    stddevs: Optional[List[float]] = None,
    num_trials: int = 10,
    device: str = "cuda",
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """
    Compute the local energy of a configuration in weight space by evaluating the loss and accuracy
    of the classifier with perturbed weights.

    Args:
        classifier: Neural network classifier
        dataloader: DataLoader containing the evaluation dataset
        num_samples: Number of samples to use for evaluation (None for all)
        loss_fn: Loss function (defaults to CrossEntropyLoss if None)
        stddevs: List of standard deviations for weight perturbations
        num_trials: Number of random trials per noise level
        device: Device to run computations on

    Returns:
    Dictionary containing loss and accuracy statistics for each noise level (keys are stddevs).
    dict[stddev] is a dictionary with the following keys:
        - losses: List of losses for each trial
        - accuracies: List of accuracies for each trial
        - mean_loss: Mean loss across trials
        - std_loss: Standard deviation of loss across trials
        - mean_acc: Mean accuracy across trials
        - std_acc: Standard deviation of accuracy across trials
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if stddevs is None:
        stddevs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    classifier = classifier.to(device)
    classifier.eval()
    reference_state = copy.deepcopy(classifier.state_dict())

    results = defaultdict(dict)
    for stddev in tqdm(stddevs, desc="Computing Local Energy for various noise levels"):
        stddev = round(stddev, 2)
        trial_losses, trial_accuracies = [], []
        for trial in range(num_trials):
            classifier.load_state_dict(reference_state)
            perturb_weights(classifier, stddev)

            total_loss, correct, total, batches_processed = 0, 0, 0, 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    if num_samples is not None and total >= num_samples:
                        break
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = classifier(inputs)
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)
                    batches_processed += 1

            avg_loss = total_loss / batches_processed
            accuracy = 100.0 * correct / total
            trial_losses.append(avg_loss)
            trial_accuracies.append(accuracy)
            logging.info(
                f"Trial {trial + 1}/{num_trials}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%"
            )

        results[stddev]["losses"] = trial_losses
        results[stddev]["accuracies"] = trial_accuracies
        results[stddev]["mean_loss"] = np.mean(trial_losses)
        results[stddev]["std_loss"] = np.std(trial_losses)
        results[stddev]["mean_acc"] = np.mean(trial_accuracies)
        results[stddev]["std_acc"] = np.std(trial_accuracies)
        logging.info(f"\nNoise level σ = {stddev}:")
        logging.info(
            f"Mean Loss: {results[stddev]['mean_loss']:.4f} ± {results[stddev]['std_loss']:.4f}"
        )
        logging.info(
            f"Mean Accuracy: {results[stddev]['mean_acc']:.2f}% ± {results[stddev]['std_acc']:.2f}%"
        )

    classifier.load_state_dict(reference_state)
    return dict(results)


def plot_local_energy(results: dict, figsize=(15, 8), num_samples=None):
    """
    Plot the results from compute_local_energy function.

    Args:
        results: Dictionary output from compute_local_energy
        figsize: Tuple of (width, height) for the figure
        save_path: Optional path to save the figure

    Returns:
        matplotlib figure object
    """
    stddevs = sorted(list(results.keys()))
    mean_acc = [results[s]["mean_acc"] for s in stddevs]
    std_acc = [results[s]["std_acc"] for s in stddevs]
    mean_loss = [results[s]["mean_loss"] for s in stddevs]
    std_loss = [results[s]["std_loss"] for s in stddevs]
    num_trials = len(results[stddevs[0]]["accuracies"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.errorbar(
        stddevs,
        mean_acc,
        yerr=std_acc / np.sqrt(num_trials),
        fmt="o-",
        capsize=5,
        color="blue",
        label=f"Mean (n={num_trials})",
    )
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Model Accuracy vs. Weight Perturbation")
    ax2.errorbar(
        stddevs,
        mean_loss,
        yerr=std_loss / np.sqrt(num_trials),
        fmt="o-",
        capsize=5,
        color="red",
        label=f"Mean (n={num_trials})",
    )
    ax2.set_ylabel("Loss")
    ax2.set_title("Model Loss vs. Weight Perturbation")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Standard Deviation (σ)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    title = f"Local Energy Analysis (n={num_trials} trials), error bar represents SEM."
    if num_samples:
        title += f" Evaluating on {num_samples} samples."
    fig.suptitle(
        title,
        y=1.05,
    )
    plt.tight_layout()

    return fig


@torch.no_grad()
def compute_input_flatness(
    classifier: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: Optional[int] = None,
    stddevs: Optional[List[float]] = None,
    num_trials: int = 10,
    device: str = "cuda",
):
    """
    Compute the local energy of an input by evaluating the model's predictions
    on perturbed versions of the input using multiplicative Gaussian noise.

    Args:
        classifier: Neural network classifier
        dataloader: DataLoader containing the evaluation dataset
        num_samples: Number of samples to use for evaluation (None for all)
        stddevs: List of standard deviations for input perturbations
        num_trials: Number of random trials per noise level
        device: Device to run computations

    Returns:
    Dictionary containing prediction statistics for each noise level (keys are stddevs), and ground truth labels.
    results[stddev] is a dictionary with the following keys:
        - logits: List of logits tensors for each trial. shape num_trials x batch_size x num_classes
        - target_probas: List of probabilities for target class. shape num_trials x batch_size
    """
    if stddevs is None:
        stddevs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    classifier = classifier.to(device).eval()

    results = defaultdict(lambda: {"logits": [], "target_probas": []})
    processed_samples = 0
    all_y = []

    for batch_x, batch_y in tqdm(dataloader, desc="Processing batches"):
        all_y.append(batch_y)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        for i, stddev in enumerate(stddevs):
            stddev = round(stddev, 2)
            trial_logits, trial_probas = [], []

            for _ in range(num_trials):
                noise = torch.randn_like(batch_x) * stddev
                perturbed_image = batch_x * (1.0 + noise)
                logits = classifier(perturbed_image)
                probas = torch.softmax(logits, dim=1)
                target_probas = probas[torch.arange(len(probas)), batch_y]
                trial_logits.append(logits.cpu())
                trial_probas.append(target_probas.cpu())

            results[stddev]["logits"].append(torch.stack(trial_logits, dim=0))
            results[stddev]["target_probas"].append(torch.stack(trial_probas, dim=0))

        processed_samples += len(batch_x)
        if num_samples and processed_samples >= num_samples:
            break

    for stddev in results:
        results[stddev]["logits"] = torch.cat(results[stddev]["logits"], dim=1)
        results[stddev]["target_probas"] = torch.cat(
            results[stddev]["target_probas"], dim=1
        )
    ground_truth = torch.cat(all_y, dim=0)
    return dict(results), ground_truth


def _filter_misclassified(results: dict, ground_truth: torch.Tensor):
    """
    Filter out misclassified samples from results and data. Use the probabilities
    in results[0.0]["target_probas"] to determine misclassification.
    """
    target_probas = results[min(results.keys())]["target_probas"][
        0
    ]  # first trial, min noise
    correct_mask = target_probas > 0.5  # batch_size,
    ground_truth = ground_truth[correct_mask]

    filtered_results = {}
    for noise in results:
        filtered_results[noise] = {
            "target_probas": results[noise]["target_probas"][:, correct_mask],
            "logits": results[noise]["logits"][:, correct_mask, :],
        }

    return filtered_results, ground_truth


def _plot_individual_samples(
    noise_levels: list,
    results: dict,
    y: torch.Tensor,
    num_samples: int,
    num_trials: int,
    class_names: Optional[list] = None,
    figsize: tuple[int, int] = (15, 8),
):
    """Plot individual sample trajectories."""
    fig, ax = plt.subplots(figsize=figsize)
    classes = [int(idx) for idx in torch.unique(y)]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    color_idx_map = {class_idx: i for i, class_idx in enumerate(classes)}
    sample_means = torch.zeros(num_samples, len(noise_levels))
    sample_stds = torch.zeros(num_samples, len(noise_levels))

    for i, noise in enumerate(noise_levels):
        probs = results[noise]["probas_for_plotting"]  # (num_trials, batch_size)
        sample_means[:, i] = probs.mean(dim=0)[:num_samples]
        sample_stds[:, i] = probs.std(dim=0)[:num_samples] / (num_trials**0.5)

    for sample_idx in range(num_samples):
        label = class_names[y[sample_idx]] if class_names else f"Class {y[sample_idx]}"
        color = colors[color_idx_map[int(y[sample_idx])]]
        ax.errorbar(
            noise_levels,
            sample_means[sample_idx].cpu(),
            yerr=sample_stds[sample_idx].cpu(),
            marker="o",
            capsize=5,
            label=label,
            color=color,
            alpha=0.5,
        )
    ax.set_title("Individual Samples")
    ax.set_xlabel("Input Noise Standard Deviation (σ)")
    ax.set_ylabel("Target Class Probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def _plot_class_averages(
    noise_levels: list,
    results: dict,
    y: torch.Tensor,
    num_samples: int,
    num_trials: int,
    class_names: Optional[list] = None,
    figsize: tuple[int, int] = (15, 8),
) -> dict:
    """Plot class-averaged trajectories and return class statistics."""
    fig, ax = plt.subplots(figsize=figsize)
    classes = [int(idx) for idx in torch.unique(y)]
    num_classes = len(classes)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    color_idx_map = {class_idx: i for i, class_idx in enumerate(classes)}
    class_stats = {idx: {"means": [], "sems": [], "count": 0} for idx in classes}

    for class_idx in classes:
        class_mask = y[:num_samples] == class_idx
        class_stats[class_idx]["count"] = class_mask.sum().item()

        for noise in noise_levels:
            probs = results[noise]["probas_for_plotting"][:, class_mask]
            mean_prob = probs.mean().item()
            sem_prob = probs.std().item() / (probs.numel() ** 0.5)
            class_stats[class_idx]["means"].append(mean_prob)
            class_stats[class_idx]["sems"].append(sem_prob)

        label = f"{class_names[class_idx] if class_names else f'Class {class_idx}'} (n={class_stats[class_idx]['count']})"
        color = colors[color_idx_map[class_idx]]
        ax.errorbar(
            noise_levels,
            class_stats[class_idx]["means"],
            yerr=class_stats[class_idx]["sems"],
            marker="o",
            capsize=5,
            label=label,
            color=color,
            linewidth=2,
        )

    ax.set_title("Class Averages")
    ax.set_xlabel("Input Noise Standard Deviation (σ)")
    ax.set_ylabel("Target Class Probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, class_stats


def _print_class_summary(noise_levels: list, class_stats: dict) -> None:
    """Print summary statistics for each class."""
    print("\nClass-Averaged Summary Statistics:")
    for class_idx, stats in class_stats.items():
        if stats["count"] == 0:
            continue
        print(f"\nClass {class_idx} (n={stats['count']}):")
        print(f"{'Noise Level':^12} | {'Mean Prob':^12} | {'SEM':^12}")
        print("-" * 40)
        for noise, mean_p, sem_p in zip(noise_levels, stats["means"], stats["sems"]):
            print(f"{noise:^12.2f} | {mean_p:^12.4f} | {sem_p:^12.4f}")


def plot_inputs_flatness(
    results: dict,
    y: torch.Tensor,
    target: int = -1,
    num_samples: Optional[int] = None,
    figsize: tuple[int, int] = (12, 5),
    class_names: Optional[list[str]] = None,
    filter_misclassified: bool = False,
    plot_individual_samples: bool = True,
    print_summary: bool = True,
):
    """Plot the results from compute_input_local_energy showing how model predictions
    change with different noise levels. The first subplot shows individual samples,
    while the second subplot shows averages grouped by class.
    :param results: Dictionary output from compute_input_local_energy
    :param y: Ground truth labels output from compute_input_local_energy"""
    if filter_misclassified:
        results, y = _filter_misclassified(results, y)
    batch_size = len(y)
    num_samples = min(num_samples or batch_size, batch_size)
    num_trials = len(list(results.values())[0]["target_probas"])
    noise_levels = sorted([float(k) for k in results.keys()])

    for noise in results:
        if target != -1:
            logits = torch.stack(results[noise]["logits"])
            results[noise]["probas_for_plotting"] = torch.softmax(logits, dim=2)[
                :, :, target
            ]
        else:
            results[noise]["probas_for_plotting"] = results[noise]["target_probas"]

    if plot_individual_samples:
        fig1 = _plot_individual_samples(
            noise_levels,
            results,
            y,
            num_samples,
            num_trials,
            class_names,
            figsize,
        )
    else:
        fig1 = None

    fig2, class_stats = _plot_class_averages(
        noise_levels,
        results,
        y,
        num_samples,
        num_trials,
        class_names,
        figsize,
    )

    title = "Flatness in Input Space - probability of"
    if target == -1:
        title += " ground truth class."
    else:
        title += f" class {class_names[target] if class_names else target}."
    if filter_misclassified:
        title += " (Misclassified samples removed)"

    for fig in [f for f in (fig1, fig2) if f is not None]:
        fig.suptitle(
            title,
            y=1.05,
        )
        fig.tight_layout()

    if print_summary:
        _print_class_summary(noise_levels, class_stats)

    return fig1, fig2


def plot_average_input_flatness(
    results: dict,
    y: torch.Tensor,
    target: int = -1,
    figsize: tuple[int, int] = (15, 8),
    filter_misclassified: bool = False,
    class_names: Optional[list[str]] = None,
) -> None:
    """Plot the results from compute_input_local_energy showing how model predictions
    change with different noise levels. This plot shows the average probability of the target
    class across all samples."""
    if filter_misclassified:
        results, _ = _filter_misclassified(results, y)

    noise_levels = sorted([float(k) for k in results.keys()])
    means = torch.zeros(len(noise_levels))
    stds = torch.zeros(len(noise_levels))

    for i, noise in enumerate(noise_levels):
        if target != -1:
            logits = torch.stack(results[noise]["logits"])
            probs = torch.softmax(logits, dim=2)[:, :, target]
        else:
            probs = results[noise]["target_probas"]

        means[i] = probs.mean()
        stds[i] = probs.std() / (probs.numel() ** 0.5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        noise_levels,
        means.cpu(),
        yerr=stds.cpu(),
        marker="o",
        capsize=5,
        color="blue",
        linewidth=2,
    )

    ax.set_xlabel("Input Noise Standard Deviation (σ)")
    ax.set_ylabel("Target Class Probability")
    ax.grid(True, alpha=0.3)

    title = "Average Flatness in Input Space - probability of"
    if target == -1:
        title += " ground truth class."
    else:
        title += f" class {class_names[target] if class_names else target}."
    if filter_misclassified:
        title += " (Misclassified samples removed)"
    ax.set_title(
        title,
        y=1.05,
    )
    plt.tight_layout()
    return fig


def visualize_input_noise(x: torch.Tensor, stddevs=None, figsize=(15, 15)):
    """
    Visualize effect of multiplicative Gaussian noise on batch of images.

    Args:
        x: numpy array or torch tensor of shape (batch_size, channels, height, width)
        stddevs: list of noise standard deviations to try
        figsize: tuple of figure dimensions
    """
    if stddevs is None:
        stddevs = np.linspace(0.0, 1.0, 11)

    batch_size = len(x)
    fig, axes = plt.subplots(2 * batch_size, len(stddevs), figsize=figsize)

    for i in range(batch_size):
        for j, std in enumerate(stddevs):
            noise = torch.randn_like(x[i]) * std
            perturbed = x[i] * (1 + noise)
            axes[2 * i, j].imshow(prepare_tensor_image_for_plot(perturbed))
            axes[2 * i, j].axis("off")
            if i == 0:
                axes[2 * i, j].set_title(f"σ={std:.2f}")
            diff = perturbed - x[i]
            axes[2 * i + 1, j].imshow(prepare_tensor_image_for_plot(diff))
            axes[2 * i + 1, j].axis("off")
            axes[2 * i + 1, j].set_title(
                f"M={diff.abs().max().item():.2f}d={diff.pow(2).mean().item():.2f}"
            )
    fig.suptitle("Effect of Multiplicative Gaussian Noise on Input Images", y=1.05)
    plt.tight_layout()
    return fig
