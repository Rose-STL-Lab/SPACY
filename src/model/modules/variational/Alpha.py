from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import math
import lightning.pytorch as pl
import numpy as np
from scipy import cluster
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from sklearn.cluster import KMeans

from src.utils.loss import create_grid, calculate_distance

# TODO: document


class Alpha(pl.LightningModule):
    """
    A class to manage the `alpha` parameter and its sampling using the Gumbel-Softmax trick.
    """

    def __init__(self, num_variates: int, num_nodes: int, disjoint: bool = True, tau_gumbel: float = 5.0):
        """
        Args:
            num_variates (int): Number of variates.
            num_nodes (int): Number of nodes.
            tau_gumbel (float): Temperature for Gumbel-Softmax sampling.
        """
        super().__init__()
        self.tau_gumbel = tau_gumbel
        self.disjoint = disjoint # This is always true
        self.num_nodes = num_nodes
        self.num_variates = num_variates
        self.nodes_per_variate = num_nodes // num_variates
        # Initialize `alpha` as a learnable parameter 
        self.logit = nn.Parameter(torch.zeros((2, num_variates, num_nodes)), requires_grad=True)

    def sample_alpha(self) -> torch.Tensor:
        """
        Currently: 
        Assign alpha based upon disjoint variant setting

        Deprecated:
        Sample alpha using the Gumbel-Softmax trick. Returns hard samples with straight-through gradients.
        
        Returns:
            torch.Tensor: Sampled alpha tensor of shape (num_variates, num_nodes).
        """


        alpha_samples = torch.zeros(self.num_variates, self.num_nodes, device=self.logit.device)
        for i in range(self.num_variates):
            start_idx = i * self.nodes_per_variate
            end_idx = start_idx + self.nodes_per_variate
            alpha_samples[i, start_idx:end_idx] = 1

        return alpha_samples # Extract the second row (binary sampling)