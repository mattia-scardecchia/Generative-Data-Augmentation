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

    # torchvision models
    if architecture in ["resnet18", "vgg11"]:
        pretrained = config.get("pretrained", False)
        model = classifier_class(pretrained=pretrained)
        if architecture == "resnet18":
            model.fc = nn.Linear(model.fc.in_features, dataset_metadata["num_classes"])
        elif architecture == "vgg11":
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, dataset_metadata["num_classes"]
            )
        if pretrained and config.get("freeze_backbone", False):
            for name, param in model.named_parameters():
                if "fc" in name or "classifier" in name:
                    continue
                param.requires_grad = False
    else:
        model = classifier_class(config, dataset_metadata)

    return model
