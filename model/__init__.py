from torchvision import models
import torch.nn as nn

from .mlp import LinearClassifier, MLP
from .resnet9 import ResNet9


MODEL_REGISTRY = {
    "linear": LinearClassifier,
    "mlp": MLP,
    "resnet9": ResNet9,
    "resnet18": models.resnet18,
    "vgg11": models.vgg11,
}


def create_model(architecture, config):
    """
    Factory function to create a model based on architecture name and config.
    """
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}")
    model_class = MODEL_REGISTRY[architecture]

    # model-specific configurations
    if architecture in ["resnet18", "vgg11"] and config.get("pretrained", True):
        model = model_class(pretrained=True)

        if architecture == "resnet18":
            model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
        elif architecture == "vgg11":
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, config["num_classes"]
            )
    else:
        model = model_class(config)

    return model
