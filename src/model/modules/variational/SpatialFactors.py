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

class SpatialFactors(pl.LightningModule):
    """
    This class generates a matrix W of shape (num_variates, num_nodes, nx*ny), where each column is a one-hot vector.
    Sampling is performed with `torch.gumbel_softmax(..., hard=True)` to give binary samples and a straight-through gradient estimator.
    """

    def __init__(self,
                 num_variates: int,
                 num_nodes: int,
                 nx: int,
                 ny: int,
                 tau_gumbel: float = 1.0,
                 spherical: bool = False,
                 simple: bool = False):
        """
        Args:
            num_nodes: The number of nodes.
            nx: The dimension of x.
            ny: The dimension of y.
            tau_gumbel: Temperature used for gumbel softmax sampling.
        """
        super().__init__()
        self.spherical = spherical
        self.simple = simple # How many parameters we use to model the variance
        
        ## Spherical Parameters
        if self.spherical == True or self.simple == True:
            self.rho_mu = nn.Parameter(torch.zeros(
                (num_variates, num_nodes, 1, 2)), requires_grad=True)
            self.rho_logvar = nn.Parameter(torch.zeros(
                (num_variates, num_nodes, 1, 2)), requires_grad=True)
            self.gamma_mu = nn.Parameter(torch.zeros(
                (num_variates, num_nodes, 1, 1)), requires_grad=True)
            self.gamma_logvar = nn.Parameter(torch.zeros(
                (num_variates, num_nodes, 1, 1)), requires_grad=True)

        ## Stochastic Parameters
        else: 
            self.rho_mu = nn.Parameter(torch.zeros(
                (num_variates, num_nodes, 1, 2)), requires_grad=True)
            self.rho_logvar = nn.Parameter(torch.zeros(
                (num_variates, num_nodes, 1, 2)), requires_grad=True)
            
            self.gamma_mu = nn.Parameter(torch.zeros(
                (num_variates, num_nodes, 1, 6)), requires_grad=True)
            self.gamma_logvar = nn.Parameter(torch.zeros(
                (num_variates, num_nodes, 1, 6)), requires_grad=True)

        # initialize the parameters
        self.num_nodes = num_nodes
        self.nx = nx
        self.ny = ny
        self.num_variates = num_variates
        self.tau_gumbel = tau_gumbel

        self.grid_coords = create_grid(self.nx, self.ny)[None, None, ...].expand(
            self.num_variates, self.num_nodes, -1, -1)

        # sigmoid function to ensure the centers are in the range (0, 1)
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
        """ Sample centers and scale parameters for the RBF kernel.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled centers and scale parameters.
        """

        # sample centers and scale parameters
        centers = self.reparameterize(self.rho_mu, self.rho_logvar)
        centers = self.sigmoid(centers)
        scale = self.reparameterize(self.gamma_mu, self.gamma_logvar)

        if self.spherical or self.simple:
            scale = torch.exp(scale)
            return centers, scale
        
        # get the covariance matrix
        A = scale[...,:4].view(self.num_variates, -1, 2, 2)
        B = scale[...,4:].view(self.num_variates, -1, 2)
        # scale = torch.bmm(A, A.transpose(2, 3)) + torch.diag_embed(torch.exp(B))
        scale = torch.matmul(A, A.transpose(-1, -2)) + torch.diag_embed(torch.exp(B))
        # print("A",  A)
        # print("AAT", torch.bmm(A, A.transpose(1, 2)))
        # print("Centers", centers)
        # print("Scale", scale)

        return centers, scale

    def get_spatial_factors(self):
        """ Compute the spatial factors for the RBF kernel.

        Returns:
            torch.Tensor: Spatial factors of shape (num_variates, num_nodes, nx*ny).
        """
        centers, scale = self.get_centers_and_scale()

        # choose distance representation
        distance_mode = 'Haversine' if self.spherical else 'Euclidean'
        grid_dist = calculate_distance(self.grid_coords.to(self.device), centers, distance_mode=distance_mode)

        # compute the rbf kernel based on parameters/co-variance matrix
        if self.simple:
            exponent = torch.sum(-torch.square(self.grid_coords.to(self.device) - centers)/scale.expand(-1,-1,-1,2), dim=-1)
        else:
            exponent = -0.5 * torch.einsum('...ik,...kl,...il->...i', grid_dist, scale, grid_dist)
        spatial_factor = torch.exp(exponent)
        

        return spatial_factor

    
    
    def calculate_entropy(self):
        """ Calculate the entropy of the variational distribution.
        
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
        """ Compute the KL divergence between two Gaussian distributions.

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
        """ Compute the KL divergence between the variational distribution and the prior.
        
        Returns:
            torch.Tensor: Total KL divergence.
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
