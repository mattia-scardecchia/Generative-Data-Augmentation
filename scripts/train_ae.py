import yaml
from src.models.autoencoding.autoencoder import Autoencoder

config = yaml.safe_load(open("configs/mlpae.yaml", "r"))
model = Autoencoder(config)
print(model)