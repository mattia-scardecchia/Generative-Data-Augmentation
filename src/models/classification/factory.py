import torch.nn as nn
from torchvision import models

from .mlp import MLP, LinearClassifier
from .resnet9 import ResNet9

MODEL_REGISTRY = {
    "linear": LinearClassifier,
    "mlp": MLP,
    "resnet9": ResNet9,
    "resnet18": models.resnet18,
    "vgg11": models.vgg11,
}


def create_classifier(architecture, config, dataset_metadata) -> nn.Module:
    """
    Factory function to create a model based on architecture name and config.
    """
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}")
    classifier_class = MODEL_REGISTRY[architecture]

    # model-specific configurations
    if architecture in ["resnet18", "vgg11"] and config.get("pretrained", True):
        model = classifier_class(pretrained=True)

        if architecture == "resnet18":
            model.fc = nn.Linear(model.fc.in_features, dataset_metadata["num_classes"])
        elif architecture == "vgg11":
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, dataset_metadata["num_classes"]
            )
    else:
        model = classifier_class(config, dataset_metadata)

    return model
