import lightning.pytorch as pl
from torch import nn
import torch

import pyro.distributions as distrib
from pyro.distributions.transforms.spline import ConditionalSpline
from pyro.distributions import constraints
from pyro.distributions.torch_transform import TransformModule


class TemporalConditionalSplineFlow(pl.LightningModule):

    def __init__(self,
                 hypernet
                 ):
        super().__init__()

        self.hypernet = hypernet
        self.num_bins = self.hypernet.num_bins
        self.order = self.hypernet.order

    def log_prob(self,
                 X_input: torch.Tensor,
                 X_history: torch.Tensor,
                 A: torch.Tensor,
                 embeddings: torch.Tensor = None):
        """
        Args:
            X_input: input data of shape (batch, num_nodes)
            X_history: input data of shape (batch, lag, num_nodes)
            A: adjacency matrix of shape (batch, lag+1, num_nodes, num_nodes)
            embeddings: embeddings (batch, lag+1, num_nodes, embedding_dim)
        """

        # assert len(X_history.shape) == 4

        _,_, num_nodes, _ = X_history.shape

        # if not self.trainable_embeddings:
        transform = nn.ModuleList(
            [
                ConditionalSpline(
                    self.hypernet, input_dim=num_nodes, count_bins=self.num_bins, order=self.order, bound=5.0
                )
                # AffineDiagonalPyro(input_dim=self.num_nodes*self.data_dim),
                # Spline(input_dim=self.num_nodes*self.data_dim, count_bins=self.num_bins, order="quadratic", bound=5.0),
                # AffineDiagonalPyro(input_dim=self.num_nodes*self.data_dim),
                # Spline(input_dim=self.num_nodes*self.data_dim, count_bins=self.num_bins, order="quadratic", bound=5.0),
                # AffineDiagonalPyro(input_dim=self.num_nodes*self.data_dim)
            ]
        )
        base_dist = distrib.Normal(
            torch.zeros(num_nodes, device=self.device), torch.ones(
                num_nodes, device=self.device)
        )
        # else:
        context_dict = {"X": X_history, "A": A, "embeddings": embeddings}
        flow_dist = distrib.ConditionalTransformedDistribution(
            base_dist, transform).condition(context_dict)
        return flow_dist.log_prob(X_input)

    def sample(self,
               N_samples: int,
               X_history: torch.Tensor,
               A: torch.Tensor,
               embeddings: torch.Tensor):
        assert len(X_history.shape) == 4

        batch, _, num_nodes, _ = X_history.shape

        transform = nn.ModuleList(
            [
                ConditionalSpline(
                    self.hypernet, input_dim=num_nodes, count_bins=self.num_bins, order=self.order, bound=5.0
                )
            ]
        )
        base_dist = distrib.Normal(
            torch.zeros(num_nodes, device=self.device), torch.ones(
                num_nodes, device=self.device)
        )

        context_dict = {"X": X_history, "A": A, "embeddings": embeddings}
        flow_dist = distrib.ConditionalTransformedDistribution(
            base_dist, transform).condition(context_dict)
        return flow_dist.sample([N_samples, batch])

    def calculate_likelihood(self, X_true: torch.Tensor, X_predict: torch.Tensor, X_history: torch.Tensor, expanded_G = None, mean = False):
        """
        Args:
            X_true: (batch, num_node, )
            X_predict: (batch, num_nodes)
            X_history: (batch, lag, num_nodes)
        """
        batch, num_nodes = X_true.shape
        if mean:
            log_prob = self.log_prob(X_input=(X_true - X_predict).view(batch, num_nodes), 
                                    X_history=X_history.unsqueeze(-1), A=expanded_G).mean(-1)
        else:
            log_prob = self.log_prob(X_input=(X_true - X_predict).view(batch, num_nodes), 
                                    X_history=X_history.unsqueeze(-1), A=expanded_G).sum(-1)

        
        likelihood_term = -torch.mean(log_prob)
        return likelihood_term

        