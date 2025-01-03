import copy
from collections import defaultdict
from typing import Optional

import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from src.eval.flatness import perturb_weights
from src.utils import prepare_tensor_image_for_plot

# TODO: test new functions and remove deprecated ones


def compute_proba_grads_wrt_data(
    classifier: nn.Module,
    data: torch.Tensor,
    targets: list[int],
    logit_transform=None,
    device=None,
    autoencoder=None,
    epsilon: float = 1e-3,  # ignored if autoencoder is not provided
):
    """
    :param data: tensor of shape (batch_size, *data.shape) containing the input samples.
    :param autoencoder: if provided, work in latent space.

    For each pair of target and input sample, do the following:
    - if autoencoder is not provided, compute the gradient of the classifier probability
      for the target class with respect to the input data.
    - if autoencoder is provided: embed the sample, compute the gradient of the classifier
      probability for the target class with respect to the latent embedding, perturb the
      embedding in the direction of the gradient, decode the perturbed embedding, compute
      the difference with reconstruction of the original image, normalize by the
      (latent space... but what else to do?) step size.
    Note: there is spurious accumulation of gradients in classifier and autoencoder.

    :return: return a tuple of dictionaries (grads, finite_diffs) with target idxs as keys:
    - grads[idx] is a tensor of shape (batch_size, *latent.shape) containing the gradients
    of all embeddings for target class idx.
    - finite_diffs[idx] is a tensor of shape (batch_size, *data.shape) containing the finite
    differences of all samples for target class idx.
    """
    if logit_transform is None:
        logit_transform = lambda x: nn.functional.softmax(x, dim=1)
    if device is None:
        device = classifier.device
    classifier = classifier.to(device).eval()
    data = data.to(device)
    if autoencoder is None:
        grads = _compute_grads_no_autoencoder(
            classifier, data, targets, logit_transform
        )
        finite_diffs = None
    else:
        autoencoder = autoencoder.to(device).eval()
        grads, finite_diffs = _compute_grads_with_autoencoder(
            classifier, autoencoder, data, targets, logit_transform, epsilon
        )
    return grads, finite_diffs


def _compute_grads_no_autoencoder(
    classifier: nn.Module, data: torch.Tensor, targets: list[int], logit_transform
) -> dict[int, torch.Tensor]:
    data.requires_grad = True
    logits = classifier(data)
    obj = logit_transform(logits)
    grads = {}
    for target in targets:
        objective = -obj[:, target].sum()
        objective.backward(retain_graph=True)
        grads[target] = data.grad.cpu().detach()
        data.grad = None
    return grads


def _compute_grads_with_autoencoder(
    classifier: nn.Module,
    autoencoder: nn.Module,
    data: torch.Tensor,
    targets: list[int],
    logit_transform,
    epsilon: float,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    with torch.no_grad():
        latent = autoencoder.encode(data)
    latent.requires_grad = True
    data_hat = autoencoder.decode(latent)
    logits = classifier(data_hat)
    obj = logit_transform(logits)
    grads, finite_diffs = {}, {}
    for target in targets:
        objective = -obj[:, target].sum()
        objective.backward(retain_graph=True)
        with torch.no_grad():
            perturbed_latent = latent + epsilon * latent.grad
            perturbed_data_hat = autoencoder.decode(perturbed_latent)
            delta = perturbed_data_hat - data_hat
        finite_diffs[target] = delta.cpu().detach() / epsilon
        grads[target] = latent.grad.cpu().detach()
        latent.grad = None
    return grads, finite_diffs


def plot_grads_wrt_data(
    data: torch.Tensor,
    grads: dict[int, torch.Tensor],
    targets: list[int],
    class_names: list[str] = None,
    figsize=(15, 8),
):
    batch_size, num_classes = data.shape[0], len(targets)
    fig, axes = plt.subplots(
        batch_size, num_classes + 1, figsize=figsize, squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for i in range(batch_size):
        axes[i, 0].imshow(prepare_tensor_image_for_plot(data[i]))
        axes[i, 0].set_title(f"M={data[i].abs().max().item():.4f}")
        for j, target in enumerate(targets):
            axes[i, j + 1].imshow(prepare_tensor_image_for_plot(grads[target][i]))
            title = f"M={grads[target][i].abs().max().item():.4f}"
            if class_names:
                title = f"{class_names[j]}\n" + title
            axes[i, j + 1].set_title(title)
    fig.suptitle("(pesudo-)gradients of class probabilities wrt data.")
    plt.tight_layout()
    return fig


def optimize_proba_wrt_data(
    classifier: nn.Module,
    data: torch.Tensor,
    targets: list[int],
    config: dict,
    device=None,
    autoencoder=None,
):
    """
    :param data: tensor of shape (batch_size, *data.shape) containing the input samples.
    :param autoencoder: if provided, work in latent space.
    :param config: dictionary with the following keys:
    - num_steps: number of optimization steps.
    - optimizer_cls: optimizer class to use.
    - optimizer_kwargs: additional keyword arguments for the optimizer.
    - logit_transform: transformation to apply to the logits instead of softmax.
    - save_k_intermediate_imgs: save current image state every k steps.

    Optimize the input data to maximize the probability of the target classes. Do small steps
    in the direction of the gradient. If an autoencoder is provided, optimize in its latent space.

    :return: a dictionary with keys trajectory, objectives, grad_norms, final_imgs, latent_trajectory,
    final_latents. Each value is a dictionary with target idxs as keys.
    - trajectory[idx] is a list containing config["save_k_intermediate_imgs"] tuples. First element is
    an integer specifying the current step. Second is an images as a tensor of shape (batch_size, *data.shape).
    - objectives[idx] is a tensor of shape (config["num_steps"], batch_size). Each value is the
    objective (by default, the probability of the target class) at the corresponding step.
    - grad_norms[idx] is a tensor of shape (config["num_steps"], batch_size). Each value is the
    gradient l2 norm at the corresponding step.
    - final_imgs[idx] is a tensor of shape (batch_size, *data.shape) containing the final images
    after config["num_steps"] optimization steps.
    - latent_trajectory is None if autoencoder is not provided. Otherwise, latent_trajectory[idx] is
    a list containing config["save_k_intermediate_imgs"] tuples. First element is an integer specifying
    the current step. Second is a tensor of shape (batch_size, *latent.shape).
    - final_latents is None if autoencoder is not provided. Otherwise, final_latents[idx] is a tensor
    of shape (batch_size, *latent.shape) containing the final latent embeddings after
    config["num_steps"] optimization steps.
    """
    if config["optimizer_cls"] is None:
        config["optimizer_cls"] = torch.optim.SGD
    if config["logit_transform"] is None:
        config["logit_transform"] = lambda x: nn.functional.softmax(x, dim=1)
    if device is None:
        device = classifier.device
    config["save_every_k"] = (
        config["num_steps"] // config["save_k_intermediate_imgs"]
        if config["save_k_intermediate_imgs"]
        else config["num_steps"]
    )
    classifier = classifier.to(device).eval()
    data = data.to(device)
    if autoencoder is None:
        trajectory, objectives, grad_norms, final_imgs = (
            _optimize_proba_wrt_data_no_autoencoder(classifier, data, targets, config)
        )
        latent_trajectory, final_latents = None, None
    else:
        autoencoder = autoencoder.to(device).eval()
        (
            trajectory,
            objectives,
            grad_norms,
            final_imgs,
            latent_trajectory,
            final_latents,
        ) = _optimize_proba_wrt_data_with_autoencoder(
            classifier, autoencoder, data, targets, config
        )

    return {
        "trajectory": trajectory,
        "objectives": objectives,
        "grad_norms": grad_norms,
        "final_imgs": final_imgs,
        "latent_trajectory": latent_trajectory,
        "final_latents": final_latents,
    }


def _optimize_proba_wrt_data_no_autoencoder(
    classifier: nn.Module, data: torch.Tensor, targets: list[int], config: dict
):
    trajectories, objectives, grad_norms = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    final_imgs = {}
    if config["perturb_weights"]:
        reference_state = copy.deepcopy(classifier.state_dict())
    for target in targets:
        x = data.clone()
        x.requires_grad = True
        optimizer = config["optimizer_cls"]([x], **config["optimizer_kwargs"])
        trajectories[target].append((0, x.cpu().detach().clone()))
        for step in tqdm(
            range(config["num_steps"]), desc=f"Optimizing target {target}"
        ):
            if config["perturb_weights"] and step % config["weights_sample_freq"] == 0:
                classifier.load_state_dict(reference_state)
                perturb_weights(classifier, config["stddev"])
            optimizer.zero_grad()
            logits = classifier(x)
            probas = config["logit_transform"](logits)[:, target]
            (-probas.sum()).backward()  # maximize proba; batches don't interact -> sum
            optimizer.step()
            if (step + 1) % config["save_every_k"] == 0:
                trajectories[target].append((step, x.cpu().detach().clone()))
            objectives[target].append(
                probas.detach().cpu().clone()
            )  # clone not needed... but play safe
            with torch.no_grad():
                grad_norm = (x.grad**2).mean(dim=(1, 2, 3)).sqrt()
            grad_norms[target].append(grad_norm.cpu().detach().clone())
        final_imgs[target] = x.cpu().detach().clone()
    objectives = {idx: torch.stack(objs, dim=0) for idx, objs in objectives.items()}
    grad_norms = {idx: torch.stack(grads, dim=0) for idx, grads in grad_norms.items()}
    if config["perturb_weights"]:
        classifier.load_state_dict(reference_state)
    return trajectories, objectives, grad_norms, final_imgs


def _optimize_proba_wrt_data_with_autoencoder(
    classifier: nn.Module,
    autoencoder: nn.Module,
    data: torch.Tensor,
    targets: list[int],
    config: dict,
):
    trajectories, objectives, grad_norms, latent_trajectories = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    final_imgs, final_latents = {}, {}
    if config["perturb_weights"]:
        reference_state = copy.deepcopy(classifier.state_dict())
    with torch.no_grad():
        latent = autoencoder.encode(data)
    for target in targets:
        z = latent.clone()
        z.requires_grad = True
        optimizer = config["optimizer_cls"]([z], **config["optimizer_kwargs"])
        trajectories[target].append((0, data.cpu().detach().clone()))
        latent_trajectories[target].append((0, z.cpu().detach().clone()))
        for step in tqdm(
            range(config["num_steps"]), desc=f"Optimizing target {target}"
        ):
            if config["perturb_weights"] and step % config["weights_sample_freq"] == 0:
                classifier.load_state_dict(reference_state)
                perturb_weights(classifier, config["stddev"])
            optimizer.zero_grad()
            x = autoencoder.decode(z)
            logits = classifier(x)
            probas = config["logit_transform"](logits)[:, target]
            (-probas.sum()).backward()
            optimizer.step()
            if (step + 1) % config["save_every_k"] == 0:
                trajectories[target].append((step, x.cpu().detach().clone()))
                latent_trajectories[target].append((step, z.cpu().detach().clone()))
            objectives[target].append(probas.detach().cpu().clone())
            with torch.no_grad():
                grad_norm = (z.grad**2).mean(dim=(1, 2, 3)).sqrt()
            grad_norms[target].append(grad_norm.cpu().detach().clone())
        with torch.no_grad():
            final_imgs[target] = autoencoder.decode(z).cpu().detach().clone()
        final_latents[target] = z.cpu().detach().clone()
    objectives = {idx: torch.stack(objs, dim=0) for idx, objs in objectives.items()}
    grad_norms = {idx: torch.stack(grads, dim=0) for idx, grads in grad_norms.items()}
    if config["perturb_weights"]:
        classifier.load_state_dict(reference_state)
    return (
        trajectories,
        objectives,
        grad_norms,
        final_imgs,
        latent_trajectories,
        final_latents,
    )


def plot_optimization_trajectory_fixed_target(
    out: dict,
    target: int,
    class_names: Optional[list | dict] = None,
    figsizes=((15, 8), (15, 8)),
    batch_idxs: Optional[list[int]] = None,
):
    """
    Generate two figures and return them.
    - The first figure has two subplots. One shows the evolution of target probability during
        optimization for all initial samples. The other shows the evolution of the average
        gradient norm during optimization.
    - The second figure shows the optimization trajectory for all initial samples. Each row
        corresponds to a different initial sample.
    """
    probas = out["objectives"]
    grad_norms = out["grad_norms"]
    if class_names is None:
        class_names = {idx: str(idx) for idx in probas.keys()}
    num_steps = probas[target].shape[0]
    batch_size = probas[target].shape[1]
    if batch_idxs is None:
        batch_idxs = list(range(batch_size))
    fig1, axes1 = plt.subplots(1, 2, figsize=figsizes[0], squeeze=True)
    for sample_idx in batch_idxs:
        axes1[0].plot(
            range(num_steps),
            probas[target][:, sample_idx],
            label=f"sample {sample_idx}",
        )
        axes1[1].plot(
            range(num_steps),
            grad_norms[target][:, sample_idx],
            label=f"sample {sample_idx}",
        )
    for ax in axes1:
        ax.grid()
        ax.legend()
        ax.set_xlabel("Iteration")
    axes1[0].set_title("Probability of target class during optimization")
    axes1[0].set_ylabel("Target Probability")
    axes1[1].set_title("Avg gradient norm during optimization")
    axes1[1].set_ylabel("Gradient Norm")
    fig1.suptitle(f"Optimization towards target class {class_names[target]}")
    fig1.tight_layout()

    trajectory = out["trajectory"]
    num_frames = len(trajectory[target])
    fig2, axes2 = plt.subplots(
        2 * batch_size, num_frames, figsize=figsizes[1], squeeze=False
    )
    for ax in axes2.flat:
        ax.axis("off")
    for j, (step, img) in enumerate(trajectory[target]):
        for i in range(batch_size):
            axes2[2 * i, j].imshow(prepare_tensor_image_for_plot(img[i]))
            title = f"M={img[i].abs().max().item():.2f}p={probas[target][step, i].item():.2f}"
            axes2[2 * i, j].set_title(title)
            diff = img[i] - trajectory[target][0][1][i]
            axes2[2 * i + 1, j].imshow(prepare_tensor_image_for_plot(diff))
            axes2[2 * i + 1, j].set_title(f"M={diff.abs().max().item():.2f}")
    fig2.suptitle(f"Optimization trajectory for target class {class_names[target]}")
    fig2.tight_layout()
    return fig1, fig2


def plot_optimization_trajectory_fixed_sample(
    out: dict,
    sample_idx: int,
    class_names: Optional[list | dict] = None,
    figsizes=((15, 8), (15, 8)),
    targets: Optional[list[int]] = None,
):
    """
    Generate two figures and return them.
    - The first figure has two subplots. One shows the evolution of target probability during
        optimization for all target classes. The other shows the evolution of the average
        gradient norm during optimization.
    - The second figure shows the optimization trajectory for all target classes. Each row
        corresponds to a different target class.
    """
    probas = out["objectives"]
    grad_norms = out["grad_norms"]
    if class_names is None:
        class_names = {idx: str(idx) for idx in probas.keys()}
    if targets is None:
        targets = list(probas.keys())
    num_steps = probas[targets[0]].shape[0]
    fig1, axes1 = plt.subplots(1, 2, figsize=figsizes[0], squeeze=True)
    for target in targets:
        axes1[0].plot(
            range(num_steps),
            probas[target][:, sample_idx],
            label=f"target {class_names[target]}",
        )
        axes1[1].plot(
            range(num_steps),
            grad_norms[target][:, sample_idx],
            label=f"target {class_names[target]}",
        )
    for ax in axes1:
        ax.grid()
        ax.legend()
        ax.set_xlabel("Iteration")
    axes1[0].set_title("Probability of target class during optimization")
    axes1[0].set_ylabel("Target Probability")
    axes1[1].set_title("Avg gradient norm during optimization")
    axes1[1].set_ylabel("Gradient Norm")
    fig1.suptitle(f"Optimization for sample {sample_idx}")
    fig1.tight_layout()

    trajectory = out["trajectory"]
    num_frames = len(trajectory[targets[0]])
    fig2, axes2 = plt.subplots(
        2 * len(targets), num_frames, figsize=figsizes[1], squeeze=False
    )
    for ax in axes2.flat:
        ax.axis("off")
    for i, target in enumerate(targets):
        for j, (step, img) in enumerate(trajectory[target]):
            axes2[2 * i, j].imshow(prepare_tensor_image_for_plot(img[sample_idx]))
            title = f"M={img[sample_idx].abs().max().item():.2f}p={probas[target][step, sample_idx].item():.2f}"
            if j == 0:
                title = f"Target {class_names[target]}\n" + title
            axes2[2 * i, j].set_title(title)
            diff = img[sample_idx] - trajectory[target][0][1][sample_idx]
            axes2[2 * i + 1, j].imshow(prepare_tensor_image_for_plot(diff))
            axes2[2 * i + 1, j].set_title(f"M={diff.abs().max().item():.2f}")
    fig2.suptitle(f"Optimization trajectory for sample {sample_idx}")
    fig2.tight_layout()
    return fig1, fig2


def plot_optimal_images(
    out: dict,
    class_names: Optional[list | dict] = None,
    figsize=(15, 8),
):
    opt = out["final_imgs"]
    targets = list(opt.keys())
    data = out["trajectory"][targets[0]][0][1]
    probas = out["objectives"]
    if class_names is None:
        class_names = {idx: str(idx) for idx in opt.keys()}
    batch_size = len(data)
    fig, axes = plt.subplots(
        2 * batch_size, len(targets) + 1, figsize=figsize, squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for i in range(batch_size):
        axes[2 * i, 0].imshow(prepare_tensor_image_for_plot(data[i]))
        title = f"M={data[i].abs().max().item():.2f}"
        axes[2 * i, 0].set_title(title)
        for j, target in enumerate(targets):
            axes[2 * i, j + 1].imshow(prepare_tensor_image_for_plot(opt[target][i]))
            title = f"{class_names[target]}\nM={opt[target][i].abs().max().item():.2f}p={probas[target][-1][i].item():.2f}"
            axes[2 * i, j + 1].set_title(title)
            diff = opt[target][i] - data[i]
            axes[2 * i + 1, j + 1].imshow(prepare_tensor_image_for_plot(diff))
            axes[2 * i + 1, j + 1].set_title(f"M={diff.abs().max().item():.2f}")
    fig.suptitle("Optimal images for target classes.")
    plt.tight_layout()
    return fig
