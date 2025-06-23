from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import hydra
from omegaconf import DictConfig
from src.utils.pylogger import RankedLogger
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)


def run(cfg):
    log.info(f"Instantiating data factory <{cfg.data_factory._target_}>")
    data_factory = hydra.utils.instantiate(cfg.data_factory)

    data_factory.generate_data()


@hydra.main(version_base='1.3', config_path="../../configs/data_generation", config_name="synthetic.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # generate data
    run(cfg)


if __name__ == "__main__":
    main()
