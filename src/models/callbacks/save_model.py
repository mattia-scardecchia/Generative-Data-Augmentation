import torch
import io
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
        self.trained_model_name = "trained_model.pth"

    def on_train_end(self, trainer, pl_module):
        logger.info(f"saving model state dict as artifact: {self.name}")
        buffer = io.BytesIO()
        torch.save(pl_module.state_dict(), buffer)
        buffer.seek(0)
        artifact = wandb.Artifact(
            name=self.name,
            type="model",
            description=self.description,
            metadata=self.metadata,
        )
        artifact.add_file(buffer, name=self.trained_model_name)
        wandb.log_artifact(artifact)
        logger.info(f"Uploaded model with metadata as artifact: {artifact.name}")
