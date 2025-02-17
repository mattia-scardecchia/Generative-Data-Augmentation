from typing import Tuple

import torch.nn as nn

from .cae import ConvDecoder, ConvEncoder
from .mlpae import MLPDecoder, MLPEncoder


def create_autoencoder(
    architecture: str,
    config: dict,
    in_channels: int,
    input_size: int,
) -> Tuple[nn.Module, nn.Module]:
    """
    Create encoder and decoder models based on the specified architecture and config.
    """
    match config["final_nonlinearity"]:
        case "sigmoid":
            nonlinearity = nn.Sigmoid()
        case "relu":
            nonlinearity = nn.ReLU()
        case "identity":
            nonlinearity = nn.Identity()
        case _:
            value = config["final_nonlinearity"]
            raise ValueError(f"Unknown final nonlinearity: {value}")
    match architecture.lower():
        case "mlp":
            block_kwargs = config.get("block_kwargs", {})
            encoder = MLPEncoder(
                in_features=input_size,
                hidden_dims=config["hidden_dims"],
                out_features=config["latent_dim"],
                **block_kwargs,
            )
            decoder = MLPDecoder(
                in_features=config["latent_dim"],
                hidden_dims=config["hidden_dims"][::-1],
                out_features=input_size,
                **block_kwargs,
            )

        case "conv":
            block_kwargs = config.get("block_kwargs", {})
            encoder = ConvEncoder(
                in_channels=in_channels,
                hidden_dims=config["hidden_dims"],
                out_channels=config["latent_dim"],
                block_type=config["block_type"],
                downsample=config["downsample"],
                **block_kwargs,
            )
            decoder = ConvDecoder(
                in_channels=config["latent_dim"],
                hidden_dims=config["hidden_dims"][::-1],
                out_channels=in_channels,
                block_type=config["block_type"],
                upsample=config["downsample"][::-1],
                **block_kwargs,
            )

        case _:
            raise ValueError(f"Unknown architecture: {architecture}")

    return encoder, nn.Sequential(decoder, nonlinearity)
