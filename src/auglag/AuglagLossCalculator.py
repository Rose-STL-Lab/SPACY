import torch
import torch.nn as nn


class AuglagLossCalculator(nn.Module):
    def __init__(self, init_alpha: float, init_rho: float):
        super().__init__()
        self.init_alpha = init_alpha
        self.init_rho = init_rho

        self.alpha: torch.Tensor
        self.rho: torch.Tensor
        self.register_buffer("alpha", torch.tensor(
            self.init_alpha, dtype=torch.float))
        self.register_buffer("rho", torch.tensor(
            self.init_rho, dtype=torch.float))

    def forward(self, objective: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        return objective + self.alpha * constraint + self.rho * constraint * constraint / 2
