"""
Borrowed from https://github.com/microsoft/causica
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Dict

import torch
from torch.optim import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from src.auglag.AuglagLR import AuglagLR


class AuglagLRCallback(pl.Callback):
    """Wrapper Class to make the Auglag Learning Rate Scheduler compatible with Pytorch Lightning"""

    def __init__(self, scheduler: AuglagLR, log_auglag: bool = False, disabled_epochs=None):
        """
        Args:
            scheduler: The auglag learning rate scheduler to wrap.
            log_auglag: Whether to log the auglag state as metrics at the end of each epoch.
        """
        self.scheduler = scheduler
        self._log_auglag = log_auglag
        self._disabled_epochs = disabled_epochs

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        _ = trainer
        _ = batch
        _ = batch_idx
        assert isinstance(outputs, dict)

        opt = pl_module.optimizers()
        if isinstance(opt, torch.optim.Optimizer) == False:
            if (trainer.current_epoch//2) % 2 == 1: 
                optimizer = pl_module.optimizers()[1]
            else:
                optimizer = pl_module.optimizers()[0]
        else:
            optimizer = pl_module.optimizers()

        # assert isinstance(optimizer, torch.optim.Optimizer)
        auglag_loss: AuglagLossCalculator = pl_module.loss_calc  # type: ignore

        # Disable if we reached a disabled epoch - disable, otherwise make sure the scheduler is enabled
        if self._disabled_epochs!= None and (trainer.current_epoch <= self._disabled_epochs):
            self.scheduler.disable(auglag_loss)
        else:
            self.scheduler.enable(auglag_loss)
        # breakpoint()
        is_converged = self.scheduler.step(
            optimizer=optimizer,
            loss=auglag_loss,
            loss_value=outputs["loss"],
            lagrangian_penalty=outputs["dagness_penalty"],
            likelihood=outputs['likelihood'],
            graph_prior=outputs['graph_prior'],
            # graph_entropy=outputs['graph_entropy']
        )

        # Notify trainer to stop if the auglag algorithm has converged
        if is_converged:
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        _ = trainer
        if self._log_auglag:
            auglag_state = {
                "num_lr_updates": self.scheduler.num_lr_updates,
                "outer_opt_counter": self.scheduler.outer_opt_counter,
                "step_counter": self.scheduler.step_counter,
                "outer_below_penalty_tol": self.scheduler.outer_below_penalty_tol,
                "outer_max_rho": self.scheduler.outer_max_rho,
                "last_best_step": self.scheduler.last_best_step,
                "last_lr_update_step": self.scheduler.last_lr_update_step,
            }
            pl_module.log_dict(auglag_state, on_epoch=True,
                               rank_zero_only=True, prog_bar=False)
