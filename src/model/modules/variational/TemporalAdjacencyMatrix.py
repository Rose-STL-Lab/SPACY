from typing import List, Optional, Tuple

import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from src.model.modules.variational.ThreeWayGraphDist import ThreeWayGraphDist


class TemporalAdjacencyMatrix(ThreeWayGraphDist):
    """
    This class adapts the ThreeWayGraphDist s.t. it supports the variational distributions for temporal adjacency matrix.

    The principle is to follow the logic as ThreeWayGraphDist. The implementation has two separate part:
    (1) categorical distribution for instantaneous adj (see ThreeWayGraphDist); (2) Bernoulli distribution for lagged
    adj. Note that for lagged adj, we do not need to follow the logic from ThreeWayGraphDist, since lagged adj allows diagonal elements
    and does not have to be a DAG. Therefore, it is simpler to directly model it with Bernoulli distribution.
    """

    def __init__(
        self,
        input_dim: int,
        lag: int,
        tau_gumbel: float = 1.0,
        init_logits: Optional[List[float]] = None,
    ):
        """
        This creates an instance of variational distribution for temporal adjacency matrix.
        Args:
            device: Device used.
            input_dim: The number of nodes for adjacency matrix.
            lag: The lag for the temporal adj matrix. The adj matrix has the shape (lag+1, num_nodes, num_nodes).
            tau_gumbel: The temperature for the gumbel softmax sampling.
            init_logits: The initialized logits value. If None, then use the default initlized logits (value 0). Otherwise,
            init_logits[0] indicates the non-existence edge logit for instantaneous effect, and init_logits[1] indicates the
            non-existence edge logit for lagged effect. E.g. if we want a dense initialization, one choice is (-7, -0.5)
        """
        # Call parent init method, this will init a self.logits parameters for instantaneous effect.
        super().__init__(input_dim=input_dim, tau_gumbel=tau_gumbel)
        # Create a separate logit for lagged adj
        # The logits_lag are initialized to zero with shape (2, lag, input_dim, input_dim).
        # logits_lag[0,...] indicates the logit prob for no edges, and logits_lag[1,...] indicates the logit for edge existence.
        self.lag = lag
        # Assertion lag > 0
        assert lag > 0
        self.logits_lag = nn.Parameter(torch.zeros(
            (2, lag, input_dim, input_dim), device=self.device), requires_grad=True)
        self.init_logits = init_logits
        # Set the init_logits if not None
        if self.init_logits is not None:
            self.logits.data[2, ...] = self.init_logits[0]
            self.logits_lag.data[0, ...] = self.init_logits[1]

    def get_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """
        This returns the temporal adjacency matrix of edge probability.
        Args:
            do_round: Whether to round the edge probabilities.

        Returns:
            The adjacency matrix with shape [lag+1, num_nodes, num_nodes].
        """

        # Create the temporal adj matrix
        probs = torch.zeros(self.lag + 1, self.input_dim,
                            self.input_dim, device=self.device)
        # Generate simultaneous adj matrix
        probs[0, ...] = super().get_adj_matrix(
            do_round=do_round)  # shape (input_dim, input_dim)
        # Generate lagged adj matrix
        probs[1:, ...] = F.softmax(self.logits_lag, dim=0)[
            1, ...]  # shape (lag, input_dim, input_dim)
        if do_round:
            return probs.round()
        else:
            return probs

    def entropy(self) -> torch.Tensor:
        """
        This computes the entropy of the variational distribution. This can be done by (1) compute the entropy of instantaneous adj matrix(categorical, same as ThreeWayGraphDist),
        (2) compute the entropy of lagged adj matrix (Bernoulli dist), and (3) add them together.
        """
        # Entropy for instantaneous dist, call super().entropy
        entropies_inst = super().entropy()

        # Entropy for lagged dist
        # batch_shape [lag], event_shape [num_nodes, num_nodes]

        dist_lag = td.Independent(td.Bernoulli(
            logits=self.logits_lag[1, ...] - self.logits_lag[0, ...]), 2)
        entropies_lag = dist_lag.entropy().sum()
        # entropies_lag = dist_lag.entropy().mean()

        return entropies_lag + entropies_inst

    def sample_graph(self) -> torch.Tensor:
        """
        This samples the adjacency matrix from the variational distribution. This uses the gumbel softmax trick and returns
        hard samples. This can be done by (1) sample instantaneous adj matrix using self.logits, (2) sample lagged adj matrix using self.logits_lag.
        """

        # Create adj matrix to avoid concatenation
        adj_sample = torch.zeros(
            self.lag + 1, self.input_dim, self.input_dim, device=self.device
        )  # shape (lag+1, input_dim, input_dim)f

        # Sample instantaneous adj matrix
        adj_sample[0, ...] = self._triangular_vec_to_matrix(
            F.gumbel_softmax(self.logits, tau=self.tau_gumbel,
                             hard=True, dim=0)
        )  # shape (input_dim, input_dim)
        # Sample lagged adj matrix
        adj_sample[1:, ...] = F.gumbel_softmax(self.logits_lag, tau=self.tau_gumbel, hard=True, dim=0)[
            1, ...
        ]  # shape (lag, input_dim, input_dim)
        return adj_sample

    def calculate_sparsity(self, G):
        return torch.sum(G)

    def calculate_dagness_penalty(self, G):
        num_nodes = G.shape[-1]
        trace_term = torch.trace(torch.matrix_exp(G))
        return (trace_term - num_nodes)
