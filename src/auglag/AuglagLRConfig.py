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


@dataclass
class AuglagLRConfig:
    """Configuration parameters for the AuglagLR scheduler.

    Attributes:
        lr_update_lag: Number of iterations to wait before updating the learning rate.
        lr_update_lag_best: Number of iterations to wait after the best model before updating the learning rate.
        lr_init_dict: Dictionary of intitialization parameters for every new inner optimization step. This must contain
            all parameter_groups for all optimizers
        aggregation_period: Aggregation period to compare the mean of the loss terms across this period.
        lr_factor: Learning rate update schedule factor (exponential decay).
        penalty_progress_rate: Number of iterations to wait before updating rho based on the dag penalty.
        safety_rho: Maximum rho that could be updated to.
        safety_alpha: Maximum alpha that could be udated to.
        max_lr_down: Maximum number of lr update times to decide inner loop termination.
        inner_early_stopping_patience: Maximum number of iterations to run after the best inner loss to terminate inner
            loop.
        max_outer_steps: Maximum number of outer update steps.
        patience_penalty_reached: Maximum number of outer iterations to run after the dag penalty has reached a good
            value.
        patience_max_rho: Maximum number of iterations to run once rho threshold is reached.
        penalty_tolerance: Tolerance of the dag penalty
        max_inner_steps: Maximum number of inner loop steps to run.
        force_not_converged: If True, it will not be reported as converged until max_outer_steps is reached.
    """

    lr_update_lag: int = 500
    lr_update_lag_best: int = 250
    lr_init_dict: Dict[str, float] = field(
        default_factory=lambda: {"vardist": 1e-2,
                                 "functional_relationships": 1e-2,
                                 "noise_dist": 1e-3,
                                 "causal_mapping": 1e-2,
                                 "causal_aggregation": 1e-2,
                                 "causal_deaggregation": 1e-2}
    )
    aggregation_period: int = 20
    lr_factor: float = 0.1
    penalty_progress_rate: float = 0.65
    safety_rho: float = 1e13
    safety_alpha: float = 1e13
    max_lr_down: int = 3
    init_rho: float = 1
    init_alpha: float = 0
    inner_early_stopping_patience: int = 500
    max_outer_steps: int = 100
    patience_penalty_reached: int = 5
    patience_max_rho: int = 3
    penalty_tolerance: float = 1e-5
    max_inner_steps: int = 3000
    force_not_converged: bool = False
