import os
import sys
import logging
import warnings

# Make console output UTF-8 friendly on Windows so logging messages with emoji
# do not crash the run with UnicodeEncodeError.
for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Keep external libraries quiet before they are imported.
os.environ.setdefault("WANDB_SILENT", "true")

logging.basicConfig(level=logging.WARNING, format="%(message)s")
for name in (
    "httpx",
    "datasets",
    "transformers",
    "huggingface_hub",
    "wandb",
    "pytorch_lightning",
    "lightning",
    "lightning_utilities",
    "lightning_utilities.core.rank_zero",
    "lightning_fabric",
    "urllib3",
    "filelock",
    "pytorch_lightning.utilities.rank_zero",
):
    logging.getLogger(name).setLevel(logging.WARNING)

# Suppress non-actionable warnings from dependencies.
warnings.filterwarnings(
    "ignore",
    message="triton not found; flop counting will not work for triton kernels",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"pytorch_lightning\.utilities\._pytree",
)
warnings.filterwarnings(
    "ignore",
    message="does not have many workers which may be a bottleneck",
    module=r"pytorch_lightning\.trainer\.connectors\.data_connector",
)
warnings.filterwarnings(
    "ignore",
    message=r"Artifact .* already exists with the same content\. No new version will be created\.",
    module=r"wandb\..*",
)

import torch
import hydra
import wandb

import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = self.datamodule.tokenizer.batch_decode(
            val_batch["input_ids"], skip_special_tokens=True
        )

        outputs = pl_module(
            val_batch["input_ids"].to(pl_module.device),
            val_batch["attention_mask"].to(pl_module.device),
        )
        preds = torch.argmax(outputs.logits, 1)
        # move tensors to CPU and convert to numpy before creating DataFrame
        preds_np = preds.detach().cpu().numpy()
        labels_tensor = val_batch["label"]
        if isinstance(labels_tensor, torch.Tensor):
            labels_np = labels_tensor.detach().cpu().numpy()
        else:
            labels_np = labels_tensor

        df = pd.DataFrame({"Sentence": sentences, "Label": labels_np, "Predicted": preds_np})

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


class EnsureTrainModeCallback(pl.Callback):
    """Ensure the LightningModule is set to train mode at the start of fitting.

    Some libraries or model components may be left in eval mode before
    training starts, which triggers a warning in PyTorch Lightning. This
    callback forces the module into train mode to avoid that warning.
    """

    def on_fit_start(self, trainer, pl_module):
        pl_module.train()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", entity="thesheep")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            EnsureTrainModeCallback(),
            SamplesVisualisationLogger(cola_data),
            early_stopping_callback,
        ],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        # limit_train_batches=cfg.training.limit_train_batches,
        # limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()
