import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_tensor_image_for_plot(img: torch.Tensor) -> np.ndarray:
    """
    Prepare an image tensor for plotting by converting to numpy array and normalizing.

    :param img: Input image tensor [C, H, W]
    :return: Prepared image as numpy array [H, W, C]
    """
    numpy_img = img.cpu().detach().numpy()

    # Ensure the image is in range [0, 1]
    if numpy_img.max() > 1.0 or numpy_img.min() < 0.0:
        normalized_img = (numpy_img - numpy_img.min()) / (
            numpy_img.max() - numpy_img.min()
        )
    else:
        normalized_img = numpy_img

    # Rearrange dimensions from [C, H, W] to [H, W, C]
    transposed_img = np.transpose(normalized_img, (1, 2, 0))

    # Convert grayscale to RGB
    if transposed_img.shape[2] == 1:
        final_img = np.repeat(transposed_img, 3, axis=2)
    else:
        final_img = transposed_img

    return final_img
