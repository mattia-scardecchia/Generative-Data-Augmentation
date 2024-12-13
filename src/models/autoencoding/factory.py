from typing import Tuple
import torch.nn as nn
from .cae import ConvEncoder, ConvDecoder
from .mlpae import MLPEncoder, MLPDecoder


def create_autoencoder(
    architecture: str,
    config: dict,
    in_channels: int,
    input_size: int,
) -> Tuple[nn.Module, nn.Module]:
    """
    Create encoder and decoder models based on the specified architecture and config.
    """
    if "hidden_dims" not in config:
        raise ValueError("Config must specify 'hidden_dims'")
    if "latent_dim" not in config:
        raise ValueError("Config must specify 'latent_dim'")
        
    match architecture.lower():
        case "mlp":
            block_kwargs = config.get("block_kwargs", {})
            
            encoder = MLPEncoder(
                in_features=input_size,
                hidden_dims=config["hidden_dims"],
                out_features=config["latent_dim"],
                **block_kwargs
            )
            
            decoder = MLPDecoder(
                in_features=config["latent_dim"],
                hidden_dims=config["hidden_dims"][::-1],  # Reverse for decoder
                out_features=input_size,
                **block_kwargs
            )
            
        case "conv":
            required_conv = ["block_type", "downsample", "upsample"]
            for key in required_conv:
                if key not in config:
                    raise ValueError(f"Conv architecture requires '{key}' in config")
            
            block_kwargs = config.get("block_kwargs", {})
            
            encoder = ConvEncoder(
                in_channels=in_channels,
                block_type=config["block_type"],
                hidden_dims=config["hidden_dims"],
                downsample=config["downsample"],
                **block_kwargs
            )
            
            decoder = ConvDecoder(
                in_channels=config["hidden_dims"][-1],
                block_type=config["block_type"],
                hidden_dims=config["hidden_dims"][::-1],
                upsample=config["upsample"][::-1],
                **block_kwargs
            )
            
        case _:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    return encoder, decoder
