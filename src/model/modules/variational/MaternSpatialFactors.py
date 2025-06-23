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


class SpatialFactors(pl.LightningModule):
    """
    SpatialFactors class using the Matern kernel with a fixed parameter nu (ablation).
    """

    def __init__(

        self,
        num_variates: int,
        num_nodes: int,
        nx: int,
        ny: int,
        nu: float = 2.5,  # Fixed parameter nu for the Matern kernel: 1.5, 2.5
    ):
        super().__init__()
        self.nu = nu  # Store nu as an attribute


        self.rho_mu = nn.Parameter(torch.zeros(
            (num_variates, num_nodes, 1, 2)), requires_grad=True)
        self.rho_logvar = nn.Parameter(torch.zeros(
            (num_variates, num_nodes, 1, 2)), requires_grad=True)
        
        self.gamma_mu = nn.Parameter(torch.zeros(
            (num_variates, num_nodes, 1, 6)), requires_grad=True)
        self.gamma_logvar = nn.Parameter(torch.zeros(
            (num_variates, num_nodes, 1, 6)), requires_grad=True)

        self.num_nodes = num_nodes
        self.nx = nx
        self.ny = ny
        self.num_variates = num_variates

        self.grid_coords = create_grid(self.nx, self.ny)[None, None, ...].expand(
            self.num_variates, self.num_nodes, -1, -1)

        self.sigmoid = nn.Sigmoid()

    def reparameterize(self,
                       mean: torch.Tensor,
                       logvar: torch.Tensor):
        """ Reparameterization trick to sample from a Gaussian distribution.

        Args:
            mean (torch.Tensor): Mean of the Gaussian distribution.
            logvar (torch.Tensor): Log variance of the Gaussian distribution.
        Returns:
            torch.Tensor: Sampled tensor from the Gaussian distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps*std + mean

    def get_centers_and_scale(self):
        """ Sample centers and scale parameters for the Matern kernel.


        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled centers and scale parameters.
        """

        # Sample centers and scale parameters
        centers = self.reparameterize(self.rho_mu, self.rho_logvar)
        centers = self.sigmoid(centers)
        scale = self.reparameterize(self.gamma_mu, self.gamma_logvar)


        # Obtain the covariance matrix
        A = scale[...,:4].view(-1, 2, 2)
        B = scale[...,4:].view(-1, 2)
        scale = torch.bmm(A, A.transpose(1, 2)) + torch.diag_embed(torch.exp(B))

        return centers, scale

    def get_spatial_factors(self):
        """Computes the spatial factors `F` using the Matern kernel with fixed nu.

        Returns:
            torch.Tensor: Spatial factors `F` of shape `(num_variates, num_nodes, nx * ny)`.
        """
        centers, scale = self.get_centers_and_scale()
        grid_diff = self.grid_coords.to(self.device) - centers

        # Compute the Mahalanobis distance r
        r_squared = torch.einsum('...ik,...kl,...il->...i', grid_diff, scale, grid_diff)
        r = torch.sqrt(r_squared + 1e-12)  # Add small epsilon to avoid sqrt(0)

        # Compute the Matern kernel based on the fixed nu
        if self.nu == 0.5:
            # Exponential kernel
            F = torch.exp(-r)
        elif self.nu == 1.5:
            sqrt_3_r = math.sqrt(3) * r
            F = (1 + sqrt_3_r) * torch.exp(-sqrt_3_r)
        elif self.nu == 2.5:
            sqrt_5_r = math.sqrt(5) * r
            F = (1 + sqrt_5_r + (5.0 / 3.0) * r_squared) * torch.exp(-sqrt_5_r)
        else:
            # For general nu, implementation of the Bessel function K_nu is required
            raise NotImplementedError(f"Matern kernel with nu={self.nu} is not implemented.")
        return F

    
    
    def calculate_entropy(self):
        """ Calculates the entropy of the variational distribution.
        
        Returns:
            torch.Tensor: Entropy of the variational distribution.
        """
        D = self.rho_logvar.shape[-1]
        rho_entropy = 0.5*(torch.sum(self.rho_logvar) + D *
                          (1 + math.log(2*math.pi)))
        D = self.gamma_logvar.shape[-1]
        gamma_entropy = 0.5*(torch.sum(self.gamma_logvar) + D *
                          (1 + math.log(2*math.pi)))
        return rho_entropy + gamma_entropy
    
    def kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        """ Computes the KL divergence between two Gaussian distributions.

        Args:
            mu_q (torch.Tensor): Mean of the variational distribution.
            logvar_q (torch.Tensor): Log variance of the variational distribution.
            mu_p (torch.Tensor): Mean of the prior distribution.
            logvar_p (torch.Tensor): Log variance of the prior distribution.
        Returns:
            torch.Tensor: KL divergence between the two distributions.
        """

        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl_div = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / var_p - 1).sum()
        return kl_div
    
    def compute_kl_divergence(self):
        """ Computes the KL divergence between the variational distribution and the prior.

        Returns:
            torch.Tensor: KL divergence between the variational distribution and the prior.
        """

        # Prior parameters for rho and gamma (assumed to be standard normal for simplicity)
        rho_prior_mu = torch.zeros_like(self.rho_mu)
        rho_prior_logvar = torch.zeros_like(self.rho_logvar)
        gamma_prior_mu = torch.zeros_like(self.gamma_mu)
        gamma_prior_logvar = torch.zeros_like(self.gamma_logvar)

        # Compute KL divergence for centers (rho)
        kl_rho = self.kl_divergence(
            self.rho_mu, self.rho_logvar, rho_prior_mu, rho_prior_logvar)

        # Compute KL divergence for scales (gamma)
        kl_gamma = self.kl_divergence(
            self.gamma_mu, self.gamma_logvar, gamma_prior_mu, gamma_prior_logvar)

        total_kl = kl_rho + kl_gamma
        return total_kl

    def logit_transform(self, y):
        """
        Computes the logit function x = ln(y / (1 - y))

        Args:
            y (torch.Tensor): Input tensor with values in the range (0, 1).

        Returns:
            torch.Tensor: Transformed tensor x.
        """
        # Ensure y is within the valid range (0, 1)
        eps = 1e-10  # Small epsilon to avoid division by zero or log of zero
        y = torch.clamp(y, eps, 1 - eps)
        
        # Compute the logit transformation
        x = torch.log(y / (1 - y))
        
        return x

    def cluster_initialize(self, X):
        """ Cluster initialization using KMeans for the spatial factors.

        Args:
            X (torch.Tensor): Input tensor of shape (batch, num_variates, lag+1, num_grid_points).
        Returns:
            None
        Deprecated: This function is not used in the current implementation.
        """

        spatial_feature = X.squeeze(1)[0].T #(grid_size, time_len)

        # Use KMeans to find initial cluster centers
        kmeans = KMeans(n_clusters=self.num_nodes, random_state=0)
        kmeans.fit(spatial_feature.detach().cpu().numpy())

        # Find labels/clusters of the grid
        km_labels = torch.tensor(kmeans.labels_)
        unique_labels = torch.unique(km_labels)

        # Find the mean/center of each cluster
        cluster_centers_indices = [torch.where(km_labels == label)[0].float().mean().item() for label in unique_labels]
        cluster_centers_2d = [(int(idx) // self.nx, int(idx) % self.nx) for idx in cluster_centers_indices]

        # Update rho_mu with the new cluster-based initialization
        rho_ini = (torch.tensor(cluster_centers_2d, device=self.device).float()/self.nx).clone()

        # Pass through inverse sigmoid
        rho_ini = self.logit_transform(rho_ini)
        self.rho_mu.data = rho_ini.unsqueeze(0).unsqueeze(2)
