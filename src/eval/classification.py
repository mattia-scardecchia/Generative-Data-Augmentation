import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.utils import prepare_tensor_image_for_plot


def inference_with_classifier(classifier, dataloader, device="cuda"):
    """
    Do batched inference with a PyTorch Lightning classifier on a provided dataloader's data.

    Args:
        classifier: PyTorch Lightning Module (the classifier model)
        dataloader: PyTorch DataLoader
        device: str, one of ['cpu', 'cuda']

    Returns:
        dict: dictionary containing the following keys: 'target', 'pred', 'logits'
    """
    classifier.to(device).eval()
    all_y = []
    all_pred = []
    all_logits = []

    with torch.no_grad():
        for x, y in tqdm(
            dataloader, desc=f"Doing inference on {len(dataloader)} batches on {device}"
        ):
            x = x.to(device)
            logits = classifier(x)
            preds = torch.argmax(logits, dim=1)

            all_y.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_logits = np.concatenate(all_logits)

    return {"target": y_true, "pred": y_pred, "logits": y_logits}


def collect_misclassified(classifier, dataloader, device, num_samples=9):
    """
    Collect misclassified examples from a classifier's predictions on a dataset.

    Args:
        classifier: PyTorch model in eval mode
        dataloader: PyTorch DataLoader containing the validation/test data
        num_samples: number of misclassified samples to collect (default=9)

    Returns:
        tuple: (misclassified_images, true_labels, predicted_labels)
        - misclassified_images: list of tensors containing misclassified images
        - true_labels: list of true class labels
        - predicted_labels: list of predicted class labels
    """
    classifier.to(device).eval()
    misclassified_images = []
    true_labels = []
    pred_labels = []
    pred_logits = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predictions = torch.max(outputs, 1)
            mask = predictions != labels
            misclassified_idx = torch.where(mask)[0]

            for idx in misclassified_idx:
                if len(misclassified_images) >= num_samples:
                    break

                misclassified_images.append(images[idx].cpu())
                true_labels.append(labels[idx].item())
                pred_labels.append(predictions[idx].item())
                pred_logits.append(outputs[idx].cpu())

            if len(misclassified_images) >= num_samples:
                break

    return misclassified_images, true_labels, pred_labels, pred_logits


def plot_image_grid(
    images: list[torch.Tensor],
    true_labels: list[int],
    pred_logitss: list[torch.Tensor],
    class_names: Optional[list[str]] = None,
):
    """
    Plot a grid of images with their true and predicted labels.

    Args:
        images: list of image tensors
        true_labels: list of true class labels
        pred_logitss: list of prediction logits tensors
        class_names: optional list of class names to use instead of indices
    """
    if len(images) != len(true_labels) or len(images) != len(pred_logitss):
        raise ValueError(
            "Number of images, true labels, and predicted logits must match"
        )
    n_images = len(images)
    grid_size = math.ceil(math.sqrt(n_images))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15), squeeze=False)
    fig.suptitle("Image Grid with Ground Truth and Predictions\n", fontsize=16)
    axes = axes.flatten()

    for idx, (img, true_label, pred_logits) in enumerate(
        zip(images, true_labels, pred_logitss)
    ):
        if idx < n_images:
            probas = torch.nn.functional.softmax(pred_logits, dim=0)
            pred_label = torch.argmax(pred_logits).item()
            pred_prob = probas[pred_label].item()
            true_prob = probas[true_label].item()

            true_text = (
                class_names[true_label] if class_names is not None else str(true_label)
            )
            pred_text = (
                class_names[pred_label] if class_names is not None else str(pred_label)
            )
            axes[idx].imshow(prepare_tensor_image_for_plot(img))

            title = f"GT: {true_text} (p = {true_prob:.2f})\nPred: {pred_text} (p = {pred_prob:.2f})"
            color = "green" if true_label == pred_label else "red"
            axes[idx].set_title(title, color=color)
        axes[idx].axis("off")

    plt.tight_layout()
    return fig
