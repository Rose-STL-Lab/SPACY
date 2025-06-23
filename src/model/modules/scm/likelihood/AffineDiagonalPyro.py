from pyro.distributions import constraints
from pyro.distributions.torch_transform import TransformModule
import torch
import torch.nn as nn

class AffineDiagonalPyro(TransformModule):
    """
    This creates a diagonal affine transformation compatible with pyro transforms
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, input_dim: int):
        super().__init__(cache_size=1)
        self.dim = input_dim
        self.a = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x: tensor with shape [batch, input_dim]
        Returns:
            Transformed inputs
        """
        return self.a.exp().unsqueeze(0) * x + self.b.unsqueeze(0)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reverse method
        Args:
            y: tensor with shape [batch, input]
        Returns:
            Reversed input
        """

        return (-self.a).exp().unsqueeze(0) * (y - self.b.unsqueeze(0))

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _ = x, y
        return self.a.unsqueeze(0)
