from .variational_autoencoder import KLAutoencoder
from .autoencoder import Autoencoder


def get_autoencoder_type(type: str, config):
    match type.lower():
        case "ae":
            return Autoencoder(config)
        case "klae":
            return KLAutoencoder(config)
        case _:
            raise ValueError(f"Unknown autoencoder type: {type}")
