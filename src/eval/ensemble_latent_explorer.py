from pathlib import Path
from typing import Mapping, Union, List, Set, Tuple
import logging

import torch
import torch.nn as nn
import wandb
import torchvision.utils as vutils

from ..models.autoencoding.autoencoder import Autoencoder, AutoencoderOutput

logger = logging.getLogger(__name__)


class EnsembleLatentExplorer:
    """
    Explorer that perturbs latent representations so that an ensemble of classifiers
    gradually changes an image’s predicted class. An image is considered to have crossed
    the boundary only if every classifier in the ensemble predicts one of the target classes.
    """

    MAX_ITER = 100

    def __init__(
        self,
        autoencoder: Autoencoder,
        save_dir: Union[str, Path],
        classifiers: List[nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        :param autoencoder: The pre-trained autoencoder.
        :param save_dir: Directory to save any outputs.
        :param classifiers: A list of identical classifiers (pre-trained on the same data).
        :param device: Device to run computations on.
        """
        self.autoencoder = autoencoder.to(device).eval()

        # Check that all classifiers share the same parameter names.
        parameters_names: List[Set[str]] = [
            {name for name, _ in classifier.named_parameters()}
            for classifier in classifiers
        ]
        assert all(
            param_names == parameters_names[0] for param_names in parameters_names
        ), "All classifiers must have the same parameter names."
        self.classifiers = nn.ModuleList(
            [classifier.to(device).eval() for classifier in classifiers]
        )
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, data: torch.Tensor, targets: List[int], lr: float = 0.001, *encode_args
    ) -> torch.Tensor:
        """
        Given a batch of images with arbitrary batch dimensions (e.g. (..., C, H, W)),
        iteratively update their latent representations until every classifier predicts one
        of the target classes.

        :param data: Input images of shape (..., C, H, W) or (C, H, W) for a single image.
        :param targets: List of target class indices.
        :param lr: Learning rate for the latent space update.
        :return: Perturbed images with the original shape.
        """
        if data.ndim < 3:
            raise ValueError("Input data must have at least 3 dimensions (C, H, W).")
        if data.ndim == 3:
            data = data.unsqueeze(0)  # Ensure at least one batch dimension

        new_data = data.clone()
        original_preds = self._compute_classifier_predictions(data)
        # Ensure that all classifiers agree on the starting class for each image.
        # All images must belong to the same class
        if not len(original_preds.unique()) == 1:
            raise ValueError(
                "Not all classifiers agree on the starting class for some images."
            )
        image_has_crossed_boundary = torch.zeros_like(original_preds[..., 0])
        for i in range(self.MAX_ITER):
            update_mask = 1 - image_has_crossed_boundary
            (
                new_data,
                delta_data,
                latent_grads,
                update_norm,
            ) = self._update_data_with_latent(new_data, targets, lr, update_mask)
            current_preds = self._compute_classifier_predictions(new_data)
            image_has_crossed_boundary = self._get_boundary_cross(
                current_preds, targets
            )
            self._log(
                new_data,
                delta_data,
                latent_grads,
                update_norm,
                image_has_crossed_boundary,
                i,
            )
            if torch.all(image_has_crossed_boundary):
                break
        else:
            logger.warning("<<Exited without convergence>>")
        return new_data

    def compute_logits(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for each classifier in the ensemble.

        :param data: Input images of shape (..., C, H, W).
        :return: Logits of shape (..., n_classifiers, n_classes).
        """
        return torch.stack(
            [classifier(data) for classifier in self.classifiers], dim=-2
        )

    @torch.no_grad()
    def _compute_classifier_predictions(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute predictions for each classifier without averaging.

        :param data: Input images of shape (..., C, H, W).
        :return: Tensor of predictions with shape (..., n_classifiers).
        """
        logits = self.compute_logits(data)  # shape: (..., n_classifiers, n_classes)
        return torch.argmax(logits, dim=-1)  # shape: (..., n_classifiers)

    def _update_data_with_latent(
        self,
        data: torch.Tensor,
        targets: List[int],
        lr: float,
        update_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Mapping[str, torch.Tensor],
        Mapping[str, Union[torch.Tensor, dict]],
        float,
    ]:
        """
        Update images by encoding them, computing gradients in the latent space, and stepping
        in the direction that increases the target probabilities.

        :param data: Input images of shape (..., C, H, W).
        :param targets: List of target class indices.
        :param lr: Learning rate for the update.
        :param update_mask: Binary mask with the same batch dims as the predictions.
        :return: A tuple of:
            - perturbed_data_hat: Updated images.
            - delta_data: Dictionary with finite-difference estimates.
            - latent_grads: Dictionary with the averaged latent gradients and per-target norms.
            - update_norm: Average norm of the latent update.
        """
        # Obtain the latent representation (without gradient tracking).
        with torch.no_grad():
            latent, *latent_elements = self.autoencoder.encode((data,))
        latent.requires_grad = True
        data_hat, *_ = self.autoencoder.decode((latent, *latent_elements))
        logits = self.compute_logits(data_hat)
        probs = self.softmax(logits)  # Shape: (..., n_classifiers, n_classes)
        grad_sum = torch.zeros_like(latent)
        grad_norms = {}
        for target in targets:
            if latent.grad is not None:
                latent.grad.zero_()
            mask = update_mask.to(latent.device).float()
            # Increase the probability for the target class.
            # We compute the loss of the ensemble
            objective = -probs[..., target].mean(dim=-2)
            # images that already fool all classifiers are not updated
            objective = (objective * mask).sum()
            objective.backward(retain_graph=True)
            grad = latent.grad.detach().clone()
            grad_norms[target] = grad.norm(p=2, dim=-1).mean().item()
            grad_sum = grad_sum + grad
        # Average gradients over targets.
        grad_avg = grad_sum / len(targets)
        update = lr * grad_avg
        update_norm = update.norm(p=2, dim=-1).mean().item()
        with torch.no_grad():
            perturbed_latent = latent + update
            perturbed_data_hat, *_ = self.autoencoder.decode(
                (perturbed_latent, *latent_elements)
            )
            finite_diff = (perturbed_data_hat - data_hat) / lr
        return (
            perturbed_data_hat,
            {"avg": finite_diff.cpu()},
            {"avg": grad_avg.cpu(), "per_target": grad_norms},
            update_norm,
        )

    def _get_boundary_cross(
        self, preds: torch.Tensor, targets: List[int]
    ) -> torch.Tensor:
        """
        Determine whether each image is fooled by every classifier.

        :param preds: Tensor of per-classifier predictions with shape (..., n_classifiers).
        :param targets: List of target class indices.
        :return: Tensor with shape matching the leading dims of preds, where a value of 1
                 indicates that all classifiers predicted a target class, and 0 otherwise.
        """
        target_tensor = torch.tensor(targets, device=preds.device)
        # Check for each classifier if its prediction is in the target set.
        is_in = torch.isin(preds, target_tensor)  # shape: (..., n_classifiers)
        return is_in.all(dim=-1).int()  # 1 if all classifiers are fooled

    def _log(
        self,
        images: torch.Tensor,
        delta_data: Mapping[str, torch.Tensor],
        latent_grads: Mapping[str, Union[torch.Tensor, dict]],
        update_norm: float,
        boundary_mask: torch.Tensor,
        iteration: int,
    ):
        """
        Logs images and metrics (e.g. gradient and update norms) to Weights & Biases.

        :param images: Tensor of images with shape (..., C, H, W).
        :param delta_data: Dictionary with finite-difference information.
        :param latent_grads: Dictionary with gradient information.
        :param update_norm: Average norm of the latent update.
        :param boundary_mask: Binary tensor with the same leading dims as predictions.
        :param iteration: Current iteration number.
        """
        metrics = {
            "iteration": iteration,
            "update_norm": update_norm,
            "boundary_rate": boundary_mask.float().mean().item(),
        }
        if "avg" in latent_grads:
            grad_avg_norm = latent_grads["avg"].norm(p=2, dim=-1).mean().item()
            metrics["grad_avg_norm"] = grad_avg_norm
        if "per_target" in latent_grads:
            for target, norm in latent_grads["per_target"].items():
                metrics[f"grad_norm_target_{target}"] = norm
        if "avg" in delta_data:
            finite_diff_norm = delta_data["avg"].norm(p=2, dim=-1).mean().item()
            metrics["finite_diff_norm_avg"] = finite_diff_norm

        # Reshape images to a 4D tensor (N, C, H, W) for logging.
        images_for_grid = images.reshape(-1, *images.shape[-3:])
        grid = vutils.make_grid(
            images_for_grid.cpu(), nrow=min(8, images_for_grid.shape[0])
        )
        wandb.log(
            {
                f"Iteration_{iteration}/images": wandb.Image(grid),
                **{f"Iteration_{iteration}/{k}": v for k, v in metrics.items()},
            }
        )
