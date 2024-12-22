import torch
from matplotlib import pyplot as plt
from torch import nn


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
    classifier: nn.Module, data: torch.Tensor, logit_transform=None, class_names=None
) -> list[torch.Tensor]:
    """
    Compute the gradient of all classifier probabilities with respect to the input data.
    Can also specify a transformation logit_transform to apply to the logits instead of softmax.
    """
    if logit_transform is None:
        logit_transform = lambda x: nn.functional.softmax(x, dim=1)  # noqa: E731
    data = data.to(classifier.device)
    data.requires_grad = True
    logits = classifier(data)
    obj = logit_transform(logits)
    grads = []
    for idx in range(logits.shape[1]):
        objective = -obj[:, idx].sum()
        objective.backward(retain_graph=True)
        grads.append(data.grad.cpu().detach())  # type: ignore
        data.grad = None

    batch_size = data.shape[0]
    num_classes = logits.shape[1]
    fig, axes = plt.subplots(batch_size, num_classes + 1, figsize=(15, 8))
    for i in range(batch_size):
        axes[i, 0].imshow(data[i].detach().cpu().numpy().transpose(1, 2, 0))
        axes[i, 0].axis("off")
        for j in range(num_classes):
            axes[i, j + 1].imshow(grads[j][i].numpy().transpose(1, 2, 0))
            axes[i, j + 1].axis("off")
            title = f"M={grads[j][i].abs().max().item():.1f}"
            # if class_names:
            #     title = f"{class_names[j]}. " + title
            axes[i, j + 1].set_title(title)

    plt.tight_layout()
    return grads


def optimize_proba_wrt_data(
    classifier: nn.Module,
    data: torch.Tensor,
    target: int,
    num_steps: int = 100,
    optimizer_cls=None,
    logit_transform=None,
    save_every_k=None,
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
    if save_every_k is None:
        save_every_k = num_steps
    data = data.to(classifier.device)
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


def plot_optimization_metrics(objectives, grad_norms, target=None):
    """
    Plot the metrics during optimization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
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
    if target:
        fig.suptitle(f"Optimization for target class {target}")
    plt.tight_layout()


def visualize_optimization_trajectory(objectives, trajectory, target=None):
    """
    Visualize the optimization trajectory.
    """
    num_frames = len(trajectory)
    num_images = objectives.shape[1]
    fig, axes = plt.subplots(num_images, num_frames, figsize=(15, 8))
    for j, (step, img) in enumerate(trajectory.items()):
        for i in range(num_images):
            axes[i, j].imshow(img[i].numpy().transpose(1, 2, 0))
            # axes[i, j].set_title(f"step {step}: prob={-objectives[step, i].item():.2f}")
            title = (
                f"M={img[i].abs().max().item():.1f}p={-objectives[step, i].item():.1f}"
            )
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
    if target:
        fig.suptitle(f"Optimization trajectory for target class {target}")
    plt.tight_layout()


def optimize_all_probas_wrt_data_and_plot(
    classifier: nn.Module,
    data: torch.Tensor,
    class_names,
    num_steps: int = 100,
    optimizer_cls=None,
    logit_transform=None,
    **optimizer_kwargs,
):
    """
    Optimize the input data to maximize the probability of all classes.
    Can also specify a transformation logit_transform to apply to the logits instead of softmax.
    """
    batch_size = data.shape[0]
    num_classes = len(class_names)
    optimized_images = []  # target_class, batch_element
    probas = []  # target_class, batch_element
    for idx in range(num_classes):
        trajectory, objectives, grad_norms = optimize_proba_wrt_data(
            classifier,
            data.clone(),
            idx,
            num_steps=num_steps,
            optimizer_cls=optimizer_cls,
            logit_transform=logit_transform,
            save_every_k=num_steps,
            **optimizer_kwargs,
        )
        optimized_images.append(trajectory[num_steps - 1])
        probas.append(objectives[num_steps - 1])

    fig, axes = plt.subplots(batch_size, num_classes + 1, figsize=(15, 8))
    for i in range(batch_size):
        axes[i, 0].imshow(data[i].numpy().transpose(1, 2, 0))
        for j in range(num_classes):
            axes[i, j + 1].imshow(optimized_images[j][i].numpy().transpose(1, 2, 0))
            axes[i, j + 1].axis("off")
            title = f"M={optimized_images[j][i].abs().max().item():.1f}p={-probas[j][i].item():.1f}"
            axes[i, j + 1].set_title(title)
    plt.tight_layout()
