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


@torch.inference_mode()
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
    for stddev in tqdm(stddevs):
        stddev = round(stddev, 2)
        trial_losses, trial_accuracies = [], []
        for trial in range(num_trials):
            classifier.load_state_dict(reference_state)
            with torch.no_grad():
                for name, param in classifier.named_parameters():
                    if "BatchNorm" in name:
                        continue
                    noise = torch.randn_like(param) * stddev
                    param.mul_(1 + noise)

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


def plot_local_energy_results(results: dict, figsize=(15, 8), num_samples=None):
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
