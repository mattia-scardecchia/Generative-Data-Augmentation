import copy
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    x: torch.Tensor,
    y: torch.Tensor,
    stddevs: Optional[List[float]] = None,
    num_trials: int = 10,
    device: str = "cuda",
):
    """
    Compute the local energy of an input by evaluating the model's predictions
    on perturbed versions of the input using multiplicative Gaussian noise.

    Args:
        classifier: Neural network classifier
        image: Input image tensor of shape (B, C, H, W)
        target_class: Target class index to track probability for
        stddevs: List of standard deviations for input perturbations
        num_trials: Number of random trials per noise level
        device: Device to run computations on

    Returns:
    Dictionary containing prediction statistics for each noise level (keys are stddevs).
    dict[stddev] is a dictionary with the following keys:
        - logits: List of logits tensors for each trial
        - target_probas: List of probabilities for target class
        - mean_prob: Mean probability of target class across trials
        - std_prob: Standard deviation of target class probability
    """
    if stddevs is None:
        stddevs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    x = x.to(device)
    classifier = classifier.to(device).eval()

    results = defaultdict(dict)
    for stddev in tqdm(
        stddevs, desc="Computing input space flatness for various noise levels"
    ):
        stddev = round(stddev, 2)
        trial_logits, trial_probas = [], []
        for trial in range(num_trials):
            noise = torch.randn_like(x) * stddev
            perturbed_image = x * (1.0 + noise)
            with torch.no_grad():
                logits = classifier(perturbed_image)
                probas = torch.softmax(logits, dim=1)
                target_probas = probas[torch.arange(len(probas)), y]
            trial_logits.append(logits.cpu())
            trial_probas.append(target_probas.cpu())

        results[stddev]["logits"] = torch.stack(
            trial_logits, dim=0
        )  # num_trials, batch_size, num_classes
        results[stddev]["target_probas"] = torch.stack(
            trial_probas, dim=0
        )  # num_trials, batch_size
        results[stddev]["mean_prob"] = np.mean(trial_probas, axis=0)  # batch_size
        results[stddev]["std_prob"] = np.std(trial_probas, axis=0)  # batch_size

    return dict(results)


def plot_input_flatness(
    results: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    target: int = -1,
    num_samples: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5),
    class_names: Optional[List[str]] = None,
    filter_misclassified: bool = False,
    print_summary: bool = True,
) -> None:
    """
    Plot the results from compute_input_local_energy showing how model predictions
    change with different noise levels. The first subplot shows individual samples,
    while the second subplot shows averages grouped by class.

    Args:
        results: Dictionary output from compute_input_local_energy
        x: Original input tensor batch that was analyzed, shape (B, C, H, W)
        y: Ground truth labels for the batch
        target: Target class to plot probabilities for. If -1, uses stored target_probas
        num_samples: Number of samples to plot. If None, uses full batch size
        figsize: Figure size for the plot
        class_names: Optional list of class names for labels
    """
    if filter_misclassified:
        original_logits = results[min(results.keys())]["logits"][
            0
        ]  # Take first trial of smallest noise
        predictions = torch.argmax(original_logits, dim=1)
        correct_mask = predictions == y
        x = x[correct_mask]
        y = y[correct_mask]

        for noise in results:
            results[noise]["target_probas"] = torch.stack(
                [p[correct_mask.cpu().numpy()] for p in results[noise]["target_probas"]]
            )
            results[noise]["logits"] = torch.stack(
                [l[correct_mask] for l in results[noise]["logits"]]
            )

        print(
            f"Filtered: kept {correct_mask.sum().item()}/{len(correct_mask)} correctly classified samples (considered predictions with stddev {min(results.keys())})"
        )

    batch_size = x.size(0)
    if num_samples is None:
        num_samples = batch_size
    num_samples = min(num_samples, batch_size)
    num_trials = len(list(results.values())[0]["target_probas"])
    num_classes = list(results.values())[0]["logits"].size(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    noise_levels = sorted([float(k) for k in results.keys()])
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

    # For individual sample plots
    sample_means = {i: [] for i in range(num_samples)}
    sample_stds = {i: [] for i in range(num_samples)}
    # For class-averaged plots
    class_means = {i: [] for i in range(num_classes)}
    class_sems = {i: [] for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}
    for i in range(num_samples):
        class_counts[y[i].item()] += 1

    for noise in noise_levels:
        if target == -1:
            probs = np.array(
                results[noise]["target_probas"]
            )  # (num_trials, batch_size)
        else:
            logits = torch.stack(results[noise]["logits"])
            probs = (
                torch.softmax(logits, dim=2)[:, :, target].cpu().numpy()
            )  # (num_trials, batch_size)

        # Calculate statistics for individual samples
        for sample_idx in range(num_samples):
            sample_probs = probs[:, sample_idx]  # (num_trials,)
            mean_prob = np.mean(sample_probs)
            std_prob = np.std(sample_probs)
            sample_means[sample_idx].append(mean_prob)
            sample_stds[sample_idx].append(std_prob)

        # Calculate statistics grouped by class
        for class_idx in range(num_classes):
            class_mask = y[:num_samples] == class_idx
            if not torch.any(class_mask):
                continue

            class_probs = probs[
                :, class_mask.cpu().numpy()
            ]  # (num_trials, num_samples_in_class)
            class_mean = np.mean(class_probs)
            class_sem = np.std(class_probs) / np.sqrt(class_probs.size)
            class_means[class_idx].append(class_mean)
            class_sems[class_idx].append(class_sem)

    # Plot 1: Individual samples
    for sample_idx in range(num_samples):
        label = class_names[y[sample_idx]] if class_names else f"Class {y[sample_idx]}"
        ax1.errorbar(
            noise_levels,
            sample_means[sample_idx],
            yerr=sample_stds[sample_idx] / np.sqrt(num_trials),
            marker="o",
            capsize=5,
            capthick=1,
            elinewidth=1,
            label=label,
            color=colors[y[sample_idx]],
            alpha=0.5,
        )
    ax1.set_title("Individual Samples")

    # Plot 2: Class-averaged
    for class_idx in range(num_classes):
        if class_counts[class_idx] == 0:
            continue

        label = class_names[class_idx] if class_names else f"Class {class_idx}"
        label = f"{label} (n={class_counts[class_idx]})"

        ax2.errorbar(
            noise_levels,
            class_means[class_idx],
            yerr=class_sems[class_idx],
            marker="o",
            capsize=5,
            capthick=1,
            elinewidth=1,
            label=label,
            color=colors[class_idx],
            linewidth=2,
        )
    ax2.set_title("Class Averages")
    ax2.legend()

    for ax in (ax1, ax2):
        ax.set_xlabel("Input Noise Standard Deviation (σ)")
        ax.set_ylabel("Target Class Probability")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Flatness in Input Space - probability of class {target if target != -1 else 'ground truth'}"
    )

    # Print summary statistics for class averages
    if print_summary:
        print("\nClass-Averaged Summary Statistics:")
        for class_idx in range(num_classes):
            if class_counts[class_idx] == 0:
                continue
            print(f"\nClass {class_idx} (n={class_counts[class_idx]}):")
            print(f"{'Noise Level':^12} | {'Mean Prob':^12} | {'SEM':^12}")
            print("-" * 40)
            for noise, mean_p, sem_p in zip(
                noise_levels, class_means[class_idx], class_sems[class_idx]
            ):
                print(f"{noise:^12.2f} | {mean_p:^12.4f} | {sem_p:^12.4f}")

    plt.tight_layout()
    return fig
