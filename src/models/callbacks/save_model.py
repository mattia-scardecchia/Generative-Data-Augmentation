import os
import torch
from typing import Dict, Optional, Any
import pytorch_lightning as pl
import logging
import wandb

logger = logging.getLogger(__name__)


class SaveModelCallback(pl.Callback):

    def __init__(
        self,
        artifact_name: str,
        artifact_metadata: Dict[str, Any],
        description: Optional[str] = None,
    ):
        super().__init__()
        self.name = artifact_name
        self.metadata = artifact_metadata
        self.description = description if description is not None else ""
        self.trained_model_path = "tmp_trained_model.pth"

    def on_train_end(self, trainer, pl_module):
        torch.save(pl_module.state_dict(), self.trained_model_path)
        artifact = wandb.Artifact(
            name=self.name,
            type="model",
            description=self.description,
            metadata=self.metadata,
        )
        artifact.add_file(self.trained_model_path, name=self.name)
        wandb.log_artifact(artifact)
        logger.info(f"Uploaded model with metadata as artifact: {artifact.name}")
        os.remove(self.trained_model_path)
        logger.info(f"Local file correctly removed")
