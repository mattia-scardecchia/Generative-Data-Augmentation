from collections import deque
from typing import Optional

from torch import nn

from src.models.classification.classifier import ImageClassifier
from src.utils import get_layers


class AdversariallyAugmentedClassifier(ImageClassifier):
    def __init__(self, config, classifier, autoencoder: Optional[nn.Module] = None):
        """
        TODO: for now, classifier and chunks point to the same memory. When saving checkpoints,
        memory is duplicated. If we simply delete self.classifier at the end of init, it will
        break the method that loads from checkpoints (init expects classifier). Need to override
        the checkpointinting methods to handle this.
        """
        super().__init__(config, classifier)

        self.autoencoder = autoencoder
        if self.autoencoder is not None:
            self.autoencoder.eval()
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        self.layer_idx = config["layer_idx"]
        self.epsilon = config["adversarial_augmentation"]["epsilon"]

        self._split_classifier()
        # del self.classifier

    def forward(self, x):
        x = self.chunk1(x)
        if self.training:  # TODO: check this
            dx = self.compute_adversarial_perturbation(x.detach())
            x = x + dx  # TODO: need to detach dx?
        x = self.chunk2(x)
        return x

    def _split_classifier(self):
        """
        Splits the classifier into two consecutive chunks.
        Data augmentation will happen at the level of the output of the first chunk.
        """
        layers = get_layers(self.classifier)
        if self.layer_idx >= len(layers) or self.layer_idx < 0:
            raise ValueError(f"Layer index {self.layer_idx} is out of range")

        self.chunk1 = nn.Sequential(*layers[: self.layer_idx])
        self.chunk2 = nn.Sequential(*layers[self.layer_idx :])

    def compute_adversarial_perturbation(self, x):
        """
        Compute the logits of the classifier for x (output of chunk1), and perturb it in the
        direction that maximizes the logit of the second highest probability class.
        If an autoencoder is provided, first encode x, then compute the perturbation similarly
        in latent space. Return the resulting perturbation in the input space after decoding.
        """
        if self.autoencoder is None:
            x.requires_grad = True
            requires_grad = deque()
            for param in self.chunk2.parameters():
                requires_grad.append(param.requires_grad)
                param.requires_grad = False
            logits = self.chunk2(x)
            second_highest = logits.argsort(dim=1)[:, -2]  # TODO: check this
            obj = logits[:, second_highest].sum()
            obj.backward()
            for param in self.chunk2.parameters():
                param.requires_grad = requires_grad.popleft()
            return x.grad * self.epsilon
        else:
            z = self.autoencoder.encode(x)
            z.requires_grad = True
            requires_grad = deque()
            for param in self.chunk2.parameters():
                requires_grad.append(param.requires_grad)
                param.requires_grad = False
            logits = self.chunk2(self.autoencoder.decode(z))
            second_highest = logits.argsort(dim=1)[:, -2]
            obj = logits[:, second_highest].sum()
            obj.backward()
            for param in self.chunk2.parameters():
                param.requires_grad = requires_grad.popleft()
            return self.autoencoder.decode(z + z.grad * self.epsilon) - x
