import torch
from torch import nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class LinearClassifier(nn.Module):
    def __init__(self, config, dataset_metadata):
        super().__init__()
        self.config = config
        self.dataset_metadata = dataset_metadata

        self.classifier = nn.Linear(dataset_metadata["input_size"], dataset_metadata["num_classes"])

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MLP(nn.Module):
    def __init__(self, config, dataset_metadata):

        super().__init__()
        self.config = config
        self.dataset_metadata = dataset_metadata
        activation = config.get("activation", "relu")

        layers = [nn.Linear(dataset_metadata["input_size"], config["hidden"][0])]
        layers.append(ACTIVATIONS[activation]())
        for i in range(1, len(config["hidden"])):
            layers.append(nn.Linear(config["hidden"][i - 1], config["hidden"][i]))
            layers.append(ACTIVATIONS[activation]())
        layers.append(nn.Linear(config["hidden"][-1], dataset_metadata["num_classes"]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)
