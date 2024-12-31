from typing import Optional

import torch
from matplotlib import pyplot as plt
from torch import nn

from src.utils import prepare_tensor_image_for_plot


def compute_proba_grad_wrt_data(
    classifier: nn.Module,
    data: torch.Tensor,
    target: int,
    logit_transform=None,
) -> torch.Tensor:
    """
    Compute the gradient of the classifier probability for the target class with respect to the input data.
    Can also specify a transformation logit_transform to apply to the logits instead of softmax.
    """
    if logit_transform is None:
        logit_transform = lambda x: nn.functional.softmax(x, dim=1)  # noqa: E731
    data = data.to(classifier.device)
    data.requires_grad = True
    logits = classifier(data)
    obj = logit_transform(logits)
    objective = obj[:, target].sum()
    objective.backward()
    return data.grad.cpu().detach()  # type: ignore


def compute_all_probas_grads_wrt_data_and_plot(
    classifier: nn.Module,
    data: torch.Tensor,
    logit_transform=None,
    device=None,
    class_names: list[str] = None,
) -> list[torch.Tensor]:
    """
    Compute the gradient of all classifier probabilities with respect to the input data.
    Can also specify a transformation logit_transform to apply to the logits instead of softmax.
    """
    if logit_transform is None:
        logit_transform = lambda x: nn.functional.softmax(x, dim=1)  # noqa: E731
    if device is None:
        device = classifier.device
    classifier = classifier.to(device).eval()
    data = data.to(device)
    data.requires_grad = True
    logits = classifier(data)
    obj = logit_transform(logits)
    grads = []
    for idx in range(logits.shape[1]):
        objective = -obj[:, idx].sum()
        objective.backward(retain_graph=True)
        grads.append(data.grad.cpu().detach())  # type: ignore
        data.grad = None

    num_classes = logits.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(logits.shape[1])]
    plot_grads_wrt_data(data, grads, num_classes, class_names)

    return grads


def plot_grads_wrt_data(data, grads, num_classes, class_names):
    """
    Plot the gradients of all classifier probabilities with respect to the input data
    for a batch of starting images.
    """
    batch_size = data.shape[0]
    fig, axes = plt.subplots(
        batch_size, num_classes + 1, figsize=(15, 8), squeeze=False
    )
    for i in range(batch_size):
        axes[i, 0].imshow(prepare_tensor_image_for_plot(data[i]))
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"M={data[i].abs().max().item():.4f}")
        for j in range(num_classes):
            axes[i, j + 1].imshow(prepare_tensor_image_for_plot(grads[j][i]))
            axes[i, j + 1].axis("off")
            title = f"M={grads[j][i].abs().max().item():.4f}"
            if class_names:
                title = f"{class_names[j]}\n" + title
            axes[i, j + 1].set_title(title)
    fig.suptitle("Gradients of probabilities wrt input data.")
    plt.tight_layout()


def optimize_proba_wrt_data_fixed_target(
    classifier: nn.Module,
    data: torch.Tensor,
    target: int,
    num_steps: int = 100,
    optimizer_cls=None,
    logit_transform=None,
    save_k_intermediate_imgs: Optional[int] = None,
    device=None,
    **optimizer_kwargs,
):
    """
    Optimize the input data to maximize the probability of the target class.
    Can also specify a transformation logit_transform to apply to the logits instead of softmax.
    """
    if optimizer_cls is None:
        optimizer_cls = torch.optim.SGD
    if logit_transform is None:
        logit_transform = lambda x: nn.functional.softmax(x, dim=1)  # noqa: E731
    if device is None:
        device = classifier.device
    save_every_k = (
        num_steps // save_k_intermediate_imgs if save_k_intermediate_imgs else num_steps
    )
    classifier = classifier.to(device).eval()
    data = data.to(device)
    data.requires_grad = True
    optimizer = optimizer_cls([data], **optimizer_kwargs)  # type: ignore

    objectives, grad_norms, trajectory = [], [], {}
    for step in range(num_steps):
        optimizer.zero_grad()
        logits = classifier(data)
        objs = -logit_transform(logits)[:, target]
        objectives.append(objs.detach().cpu().clone())
        objs.sum().backward()
        optimizer.step()
        grad_norms.append(
            (data.grad**2).mean(dim=(1, 2, 3)).sqrt().cpu().detach().clone()
        )
        if step == 0 or (step + 1) % save_every_k == 0:
            trajectory[step] = data.cpu().detach().clone()
    objectives = torch.stack(objectives, dim=0)
    grad_norms = torch.stack(grad_norms, dim=0)
    return trajectory, objectives, grad_norms


def optimize_proba_wrt_data_in_latent_space_fixed_target(
    classifier: nn.Module,
    autoencoder: nn.Module,
    data: torch.Tensor,
    target: int,
    num_steps: int = 100,
    optimizer_cls=None,
    logit_transform=None,
    save_k_intermediate_imgs: Optional[int] = None,
    device=None,
    **optimizer_kwargs,
):
    """
    Embed the input data in the latent space of the autoencoder. Optimize the latent code
    to maximize the probability of the target class. Then decode the optimized latent code.
    Can also specify a transformation logit_transform to apply to the logits instead of softmax.
    """
    if optimizer_cls is None:
        optimizer_cls = torch.optim.SGD
    if logit_transform is None:
        logit_transform = lambda x: nn.functional.softmax(x, dim=1)  # noqa: E731
    if device is None:
        device = classifier.device
    save_every_k = (
        num_steps // save_k_intermediate_imgs if save_k_intermediate_imgs else num_steps
    )
    assert (
        autoencoder.device == classifier.device
    ), f"Autoencoder and classifier must be on the same device; found {autoencoder.device} and {classifier.device}"

    classifier = classifier.to(device).eval()
    data = data.to(device)
    latent = autoencoder.encode(data)
    latent.requires_grad = True
    optimizer = optimizer_cls([latent], **optimizer_kwargs)  # type: ignore

    objectives, grad_norms, trajectory = [], [], {}
    for step in range(num_steps):
        optimizer.zero_grad()
        data_hat = autoencoder.decode(latent)
        logits = classifier(data_hat)
        objs = -logit_transform(logits)[:, target]
        objectives.append(objs.detach().cpu().clone())
        objs.sum().backward()
        optimizer.step()
        assert latent.grad is not None and latent.grad.ndim == 4
        grad_norms.append(
            (latent.grad**2).mean(dim=(1, 2, 3)).sqrt().cpu().detach().clone()
        )
        if step == 0 or (step + 1) % save_every_k == 0:
            trajectory[step] = autoencoder.decode(latent).cpu().detach().clone()
    objectives = torch.stack(objectives, dim=0)
    grad_norms = torch.stack(grad_norms, dim=0)
    return trajectory, objectives, grad_norms


def plot_optimization_metrics_fixed_target(objectives, grad_norms, target_name=None):
    """
    Plot the metrics during optimization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), squeeze=True)
    for i in range(objectives.shape[1]):
        axes[0].plot(range(objectives.shape[0]), -objectives[:, i], label=f"sample {i}")
    for i in range(grad_norms.shape[1]):
        axes[1].plot(range(grad_norms.shape[0]), grad_norms[:, i], label=f"sample {i}")
    for ax in axes:
        ax.grid()
        ax.legend()
        ax.set_xlabel("Iteration")
    axes[0].set_title("Probability of target class during optimization")
    axes[0].set_ylabel("Target Probability")
    axes[1].set_title("Avg gradient norm during optimization")
    axes[1].set_ylabel("Gradient Norm")
    if target_name:
        fig.suptitle(f"Optimization for target class {target_name}")
    plt.tight_layout()


def visualize_optimization_trajectory_fixed_target(
    objectives, trajectory, target_name=None
):
    """
    Visualize the optimization trajectory.
    """
    num_frames = len(trajectory)
    num_images = objectives.shape[1]
    fig, axes = plt.subplots(2 * num_images, num_frames, figsize=(15, 8), squeeze=False)
    for j, (step, img) in enumerate(trajectory.items()):
        for i in range(num_images):
            axes[2 * i, j].imshow(prepare_tensor_image_for_plot(img[i]))
            title = (
                f"M={img[i].abs().max().item():.2f}p={-objectives[step, i].item():.2f}"
            )
            axes[2 * i, j].set_title(title)
            axes[2 * i, j].axis("off")
            diff = trajectory[step][i] - trajectory[0][i]
            axes[2 * i + 1, j].imshow(prepare_tensor_image_for_plot(diff))
            axes[2 * i + 1, j].set_title(f"M={diff.abs().max().item():.2f}")
            axes[2 * i + 1, j].axis("off")
    if target_name:
        fig.suptitle(f"Optimization trajectory for target class {target_name}")
    plt.tight_layout()


def optimize_all_probas_wrt_data_and_plot(
    classifier: nn.Module,
    data: torch.Tensor,
    class_names: list[str],
    autoencoder: Optional[nn.Module] = None,
    num_steps: int = 100,
    optimizer_cls=None,
    logit_transform=None,
    device=None,
    **optimizer_kwargs,
):
    """
    Optimize the input data to maximize the probability of all classes.
    Can also specify a transformation logit_transform to apply to the logits instead of softmax.
    If an autoencoder is provided, optimize in its latent space.
    """
    if device is None:
        device = classifier.device
    batch_size = data.shape[0]
    num_classes = len(class_names)
    optimized_images = []  # target_class, batch_element
    probas = []  # target_class, batch_element
    for idx in range(num_classes):
        if autoencoder is None:
            trajectory, objectives, _ = optimize_proba_wrt_data_fixed_target(
                classifier,
                data.clone(),
                idx,
                num_steps=num_steps,
                optimizer_cls=optimizer_cls,
                logit_transform=logit_transform,
                save_k_intermediate_imgs=None,
                device=device,
                **optimizer_kwargs,
            )
        else:
            trajectory, objectives, _ = (
                optimize_proba_wrt_data_in_latent_space_fixed_target(
                    classifier,
                    autoencoder,
                    data.clone(),
                    idx,
                    num_steps=num_steps,
                    optimizer_cls=optimizer_cls,
                    logit_transform=logit_transform,
                    save_k_intermediate_imgs=None,
                    device=device,
                    **optimizer_kwargs,
                )
            )
        optimized_images.append(trajectory[num_steps - 1])
        probas.append(objectives[num_steps - 1])

    plot_optimal_inputs_for_probas(
        data, optimized_images, probas, batch_size, num_classes, class_names
    )


def plot_optimal_inputs_for_probas(
    data,
    optimized_images,
    probas,
    batch_size,
    num_classes,
    class_names,
):
    fig, axes = plt.subplots(
        2 * batch_size, num_classes + 1, figsize=(15, 8), squeeze=False
    )
    for i in range(batch_size):
        axes[2 * i, 0].imshow(prepare_tensor_image_for_plot(data[i]))
        axes[2 * i, 0].axis("off")
        axes[2 * i, 0].set_title(f"M={data[i].abs().max().item():.2f}")
        axes[2 * i + 1, 0].axis("off")
        for j in range(num_classes):
            axes[2 * i, j + 1].imshow(
                prepare_tensor_image_for_plot(optimized_images[j][i])
            )
            axes[2 * i, j + 1].axis("off")
            title = f"M={optimized_images[j][i].abs().max().item():.2f}p={-probas[j][i].item():.2f}"
            if class_names:
                title = f"{class_names[j]}\n" + title
            axes[2 * i, j + 1].set_title(title)
            axes[2 * i + 1, j + 1].imshow(
                prepare_tensor_image_for_plot(optimized_images[j][i] - data[i])
            )
            axes[2 * i + 1, j + 1].axis("off")
    fig.suptitle(
        "Optimal inputs for maximizing probabilities starting from images in a batch."
    )
    plt.tight_layout()
