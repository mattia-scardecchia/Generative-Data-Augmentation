import torch
from torch import nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class LinearClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.classifier = nn.Linear(config["input_size"], config["num_classes"])

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MLP(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.config = config
        activation = config.get("activation", "relu")

        layers = [nn.Linear(config["input_size"], config["hidden"][0])]
        layers.append(ACTIVATIONS[activation]())
        for i in range(1, len(config["hidden"])):
            layers.append(nn.Linear(config["hidden"][i - 1], config["hidden"][i]))
            layers.append(ACTIVATIONS[activation]())
        layers.append(nn.Linear(config["hidden"][-1], config["num_classes"]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)
