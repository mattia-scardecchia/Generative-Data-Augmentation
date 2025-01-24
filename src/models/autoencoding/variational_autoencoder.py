from typing import Literal, Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .autoencoder import Autoencoder, AutoencoderOutput


class KLAutoencoder(Autoencoder):
    """inherits from Autoencoder, redefines training step with appropriate KL loss,
    does not break api. train/validatio_test step should not be reimplemented to keep
    training logic unchanged"""

    def __init__(self, config, input_shape: Optional[tuple[int]] = None):
        super().__init__(config, input_shape)
        model_config = config["model"]["config"]
        self.kl_weight = model_config["kl_weight"]
        self.latent_to_mu = nn.Conv2d(
            model_config["latent_dim"], model_config["z_channels"], 1
        )
        self.latent_to_logvar = nn.Conv2d(
            model_config["latent_dim"], model_config["z_channels"], 1
        )
        self.z_to_latent = nn.ConvTranspose2d(
            model_config["z_channels"], model_config["latent_dim"], 1
        )

    def encode(self, x: AutoencoderOutput) -> AutoencoderOutput:
        """given x, returns z after reparametrization. also returns mu and
        logvar"""
        latent = self.encoder(x[0])
        mu = self.latent_to_mu(latent)
        logvar = self.latent_to_logvar(latent)
        z = self._reparametrize(mu, logvar)
        return (z, mu, logvar)

    def decode(self, z: AutoencoderOutput) -> AutoencoderOutput:
        """given z, mu, logvar, decodes z to x_hat, passes mu and logvar"""
        z_out, mu, logvar = z
        latent = self.z_to_latent(z_out)
        return (self.decoder(latent), mu, logvar)

    def compute_loss(
        self,
        x_hat: AutoencoderOutput,
        x: AutoencoderOutput,
        when: Literal["train", "val", "test"],
        log: bool = True,
    ) -> torch.Tensor:
        """in this case loss_args=[mu, logvar]"""
        recon_loss = self.loss_fn(x_hat[0], x[0])
        kl_loss = self._kl_divergence(mu=x_hat[1], logvar=x_hat[2])
        total_loss = recon_loss + self.kl_weight * kl_loss
        if log:
            on_step = True if when == "train" else False
            self.log(f"{when}/recon_loss", recon_loss, on_step=on_step, on_epoch=True)
            self.log(f"{when}/kl_loss", kl_loss, on_step=on_step, on_epoch=True)
            self.log(f"{when}/loss", total_loss, on_step=on_step, on_epoch=True)
        return total_loss

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """reparametrization trick. samples from a gaussian with mean mu and
        log(sigma) logvar"""
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(mu)
        return eps * std + mu

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """explicit computation for kl_divergence"""
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        return kl.sum(dim=1).mean()

    @staticmethod
    def default_config():
        ae_config = super().default_config()
        ae_config["model"]["config"]["z_channels"] = 3
        ae_config["model"]["config"]["kl_weight"] = 0.01
        return ae_config
