import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.models.classification.classifier import ImageClassifier
from src.utils import (
    compute_all_logits_grad_wrt_data,
    get_class_names,
    load_from_hydra_logs,
    optimize_data_wrt_logit,
)

hydra_path = (
    "/Users/mat/Desktop/Files/Code/Playground/outputs/prova/2024-12-20/13-59-25"
)

classifier, datamodule = load_from_hydra_logs(hydra_path, ImageClassifier)
dataloader = DataLoader(
    datamodule.test_dataset,  # inherits transforms from config
    batch_size=4,
    shuffle=True,
    num_workers=0,  # avoid issues with multiprocessing
)
print("========== Model summary ==========")
print(classifier)
print(f"Len of Dataloader: {len(dataloader)}")

x, y = next(iter(dataloader))
class_names = get_class_names("fashion_mnist")
print(f"class names: {class_names}")
batch_size = x.size(0)
num_classes = len(class_names)
logit_transform = lambda x: x
# logit_transform = lambda x: torch.nn.functional.softmax(x, dim=1)

# Gradients of logits wrt data
grads = compute_all_logits_grad_wrt_data(classifier, x, logit_transform)
grad_magnitudes = [grad.abs().mean().item() for grad in grads]
avg_grad_magnitude = np.mean(grad_magnitudes)
std_grad_magnitude = np.std(grad_magnitudes)
print(f"Average gradient magnitude: {avg_grad_magnitude}")
print(
    f"Standard deviation of average gradient magnitude across classes: {std_grad_magnitude}"
)
fig, axes = plt.subplots(batch_size, num_classes + 1, figsize=(15, 8))
for i in range(batch_size):
    axes[i, 0].imshow(np.transpose(x[i].numpy(), (1, 2, 0)))
    for j in range(num_classes):
        axes[i, j + 1].imshow(np.transpose(grads[j][i].numpy(), (1, 2, 0)))
        axes[i, j + 1].set_title(f"{class_names[j]}")
plt.tight_layout()
plt.savefig("data/figures/gradients_wrt_inputs.png")
plt.close()

# Compute images that maximize logits starting from various samples
for num_steps in [1, 3, 5, 10, 50, 100]:
    optimal_images = []
    for idx in range(num_classes):
        img = optimize_data_wrt_logit(
            classifier,
            x.clone(),
            idx,
            num_steps=num_steps,
            optimizer_cls=torch.optim.SGD,
            logit_transform=logit_transform,
            lr=(0.1 / avg_grad_magnitude),
            momentum=0.9,
        )
        optimal_images.append(img)
    fig, axes = plt.subplots(batch_size, num_classes + 1, figsize=(15, 8))
    for i in range(batch_size):
        axes[i, 0].imshow(np.transpose(x[i].numpy(), (1, 2, 0)))
        for j in range(num_classes):
            axes[i, j + 1].imshow(np.transpose(optimal_images[j][i].numpy(), (1, 2, 0)))
            axes[i, j + 1].set_title(f"{class_names[j]}")
    plt.tight_layout()
    plt.savefig(f"data/figures/optimal_inputs_for_logits-steps={num_steps}.png")
    plt.close()
