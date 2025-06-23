
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
from src.model.modules.nn.MLP import MLP

from src.utils.loss import create_grid, calculate_distance

# TODO: document


class F_encoder_dv(pl.LightningModule):
    """
    This class creates an encoder from observed to latent space (disjoint variants)
    """
    def __init__(self,
                 num_variates: int,
                 num_nodes: int,
                 lag: int,
                 nx: int,
                 ny: int,
                 temperature: float):
        """
        Args:
            num_nodes: The number of nodes.
            nx: The dimension of x.
            ny: The dimension of y.
            temperature: Temperature used for gumbel softmax sampling.
        """
        super().__init__()
        self.num_variates = num_variates
        self.num_nodes = num_nodes
        self.lag = lag
        self.nx = nx
        self.ny = ny
        self.temperature = temperature

        self.f_tilde = nn.ModuleList([
            MLP(input_dim=self.nx * self.ny,
                out_dim=2 * self.num_nodes // self.num_variates,
                hidden_dim=64,
                num_layers=2)
            for _ in range(self.num_variates)
        ])        
    
    def forward(self, X):

        batch, num_variates, lags, num_grid_points = X.shape
        Z_mean = []
        Z_logvar = []
        for i in range(num_variates):
            Z_latent = self.f_tilde[i](X[:,i])
            Z_mean.append(Z_latent[..., :self.num_nodes//self.num_variates])
            Z_logvar.append(Z_latent[..., self.num_nodes//self.num_variates:])
        Z_mean = torch.cat(Z_mean, dim = -1)
        Z_logvar = torch.cat(Z_logvar, dim = -1)
        return Z_mean, Z_logvar