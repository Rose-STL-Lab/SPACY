from networkx import number_attracting_components
import torch
import torch.nn as nn
import lightning.pytorch as pl
import pyro.distributions as distrib
from src.model.modules.scm.likelihood.AffineDiagonalPyro import AffineDiagonalPyro
from pyro.distributions.transforms.spline import Spline


class SplineLikelihood(pl.LightningModule):

    def __init__(self,
                 num_nodes: int,
                 num_bins: int = 8,
                 order: str = 'linear'):

        super().__init__()
        self.num_bins = num_bins
        self.order = order
        self.num_nodes = num_nodes
        # self.num_variates = num_variates
        # self.num_grid_points = nx*ny

    def calculate_likelihood(self, X_true: torch.Tensor, X_predict: torch.Tensor):
        """
        Args:
            X_true: (batch, num_nodes)
            X_predict: (batch, num_nodes)
        """

        batch, num_nodes = X_true.shape
        self.transform = [
            AffineDiagonalPyro(input_dim=self.num_nodes).to(self.device)
            # AffineDiagonalPyro(input_dim=grid_point*data_dim),
            # Spline(input_dim=num_grid_points, count_bins=self.num_bins,
            #        order=self.order, bound=5.0).to(self.device)
        ]
        # if not self.trainable_embeddings:
        self.base_dist = distrib.Normal(
            torch.zeros(self.num_nodes, device=self.device), torch.ones(
                self.num_nodes, device=self.device)
        )
        self.flow_dist = distrib.TransformedDistribution(
            self.base_dist, self.transform)

        return -self.flow_dist.log_prob((X_true-X_predict).view(batch, -1)).sum(-1).mean()
