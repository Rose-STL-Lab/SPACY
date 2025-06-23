from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch import seed_everything
from omegaconf import DictConfig

import hydra
from omegaconf import DictConfig
from src.utils.pylogger import RankedLogger
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.files import generate_run_name
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler

import time

log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg):
    
    start = time.time()

    run_name = generate_run_name(cfg)
    seed_everything(cfg.random_seed)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"), run_name)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, detect_anomaly=True)

    if 'PCMCI' in cfg.model._target_:
        trainer.test(model=model, datamodule=datamodule)
        return
    # fit the model
    trainer.fit(model=model, datamodule=datamodule)

    model.eval()
    trainer.test(model=model, datamodule=datamodule)

    end = time.time()

    print('Finished in '+str(round(end - start, 2))+' seconds')


@hydra.main(version_base='1.3', config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
