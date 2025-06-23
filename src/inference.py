from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from src.utils.files import generate_run_name
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers

from src.utils.pylogger import RankedLogger
import time
log = RankedLogger(__name__, rank_zero_only=True)


def infer(cfg: DictConfig) -> Any:
    """Perform inference with a pretrained model on a single batch from the datamodule.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Model outputs for the batch.
    """
    log.info("Starting inference...")
    start_time = time.time()
    run_name = generate_run_name(cfg)
    
    # Set random seed for reproducibility
    seed_everything(cfg.random_seed)

    # Instantiate datamodule and prepare data
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")  # Use appropriate stage (test/predict)

    # Instantiate model (architecture defined by config)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"), run_name)
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, detect_anomaly=True)

    # Load pretrained weights from checkpoint
    log.info(f"Loading model checkpoint from {cfg.checkpoint_path}")
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.freeze()  # Freeze all parameters
    trainer.test(model=model, datamodule=datamodule)

    log.info(f"Inference completed in {time.time() - start_time:.2f}s")


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="inference.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for inference.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    infer(cfg)


if __name__ == "__main__":
    main()