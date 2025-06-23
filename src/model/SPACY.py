from typing import Tuple

import torch
import torch.nn as nn
from src.model.modules.scm.LinearSCM import LinearSCM
from src.model.modules.scm.RhinoSCM import RhinoSCM
from src.model.modules.scm.Rhino import Rhino
from src.model.modules.variational.SpatialFactors import SpatialFactors

from src.model.modules.nn.MLP import MLP
from src.model.modules.decoding.SpatialDecoderNN import SpatialDecoderNN

from src.utils.reshaping import convert_data_to_timelagged
import lightning.pytorch as pl
from src.model.modules.variational.TemporalAdjacencyMatrix import TemporalAdjacencyMatrix
import torch.nn.functional as F
import math


class SPACY(pl.LightningModule):

    def __init__(self,
                 # model parameters
                 lag: int,
                 num_nodes: int,
                 nx: int,
                 ny: int,
                 num_variates: int,
                 scm_model: nn.Module,
                 spatial_factors: nn.Module,

                 # loss terms weightage,
                 graph_sparsity_factor: float,
                 ):
        # TODO: documentation

        super().__init__()
        # model parameters initialization
        self.lag = lag
        self.num_nodes = num_nodes
        self.num_variates = num_variates
        self.nx = nx
        self.ny = ny
        
        # encoder 
        self.f_tilde = MLP(input_dim=self.nx*self.ny,
                           out_dim=2*self.num_nodes,
                           hidden_dim=64,
                           num_layers=2)

        # spatial factors 
        self.spatial_factors = spatial_factors
        self.use_beta = True

        # latent SCM 
        self.scm_model = scm_model
        self.temporal_graph_dist = TemporalAdjacencyMatrix(input_dim=self.num_nodes,
                                                           lag=self.lag)
        self.graph_sparsity_factor = graph_sparsity_factor

        # decoder 
        self.spatial_decoder = SpatialDecoderNN(nx=self.nx,
                                                ny=self.ny,
                                                num_variates=self.num_variates,
                                                embedding_dim=32,
                                                lag=self.lag,
                                                num_nodes=self.num_nodes)

    def compute_loss_terms(self,
                           X_lag: torch.Tensor,
                           X_hat: torch.Tensor,
                           Z_mean: torch.Tensor,
                           Z_logvar: torch.Tensor,
                           Z_hat: torch.Tensor,
                           Z: torch.Tensor,
                           G: torch.Tensor,
                           total_num_fragments: int):
        """ Compute loss terms for Spacy

        Args:
            X_lag (torch.Tensor): Time lagged tensor of shape [n_fragments, num_variates, lag+1, num_grid_points]
            X_hat (torch.Tensor): Reconstructed tensor of shape [n_fragments, num_variates, num_grid_points]
            Z_mean (torch.Tensor): Mean of the latent variable
            Z_logvar (torch.Tensor): Log variance of the latent variable
            Z_hat (torch.Tensor): Reconstructed latent variable
            Z (torch.Tensor): Latent variable
            G (torch.Tensor): Graph of shape [lag+1, num_nodes, num_nodes]
            total_num_fragments (int): The total number of fragments in the training set.
        Returns:
            loss_terms (dict): Dictionary of loss terms
        """

        batch = X_lag.shape[0]
        expanded_G = G.unsqueeze(0).repeat(batch, 1, 1, 1)

        # calculate graph loss terms
        graph_sparsity = self.temporal_graph_dist.calculate_sparsity(G)
        dagness_penalty = self.temporal_graph_dist.calculate_dagness_penalty(
            G[0])/total_num_fragments

        # graph entropy
        graph_sparsity_term = self.graph_sparsity_factor * graph_sparsity
        graph_prior_term = graph_sparsity_term / total_num_fragments
        graph_entropy = -self.temporal_graph_dist.entropy() / total_num_fragments

        # get the current timestep
        X_true = X_lag[:, :, -1]

        # calculate the likelihood term
        likelihood_term = torch.sum(torch.square(X_hat - X_true)) / batch
        # cd_loss = torch.sum(torch.square(Z-Z_hat))/batch
        cd_loss = self.scm_model.calculate_likelihood(X_true=Z[:,-1,:], X_pred=Z_hat, 
                                                      X_history=Z[:,:-1,:], expanded_G = expanded_G, mean = True)
        D = Z_logvar.shape[-1]
        z_entropy = -0.5*(torch.sum(Z_logvar) + D *
                          (1 + math.log(2*math.pi))) / (batch*self.num_nodes)
        
        # calculate the KL divergence term (with or w/o adjustment)
        if self.use_beta:
            beta = self.num_nodes / 4
        else:
            beta = 1
        kl_term = beta*(cd_loss + z_entropy)
    
        # calculate the spatial factor term
        f_entropy = self.spatial_factors.calculate_entropy() / total_num_fragments
        f_term = f_entropy

        
        loss_terms = {
            'graph_sparsity': graph_sparsity_term,
            'dagness_penalty': dagness_penalty,
            'graph_entropy': graph_entropy,
            'graph_prior': graph_prior_term,
            'likelihood': likelihood_term,
            'kl_term': kl_term,
            'f_term': f_term
        }  

        return loss_terms

    def compute_loss(self,
                     X: torch.Tensor,
                     total_num_fragments: int,
                     spatial_factors: torch.Tensor = None):
        """Compute total loss and the loss terms for STCD

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, num_variates, timesteps, nx, ny]
            total_num_fragments (int): The total number of fragments in the training set.
        Returns:
            loss: total loss
            loss_terms: dictionary of loss terms
        """

        X_lag, X_hat, Z_mean, Z_logvar, Z_hat, Z, G, F = self(X, spatial_factors)

        loss_terms = self.compute_loss_terms(X_lag=X_lag,
                                             X_hat=X_hat,
                                             Z_mean=Z_mean[:, -1],
                                             Z_logvar=Z_logvar[:, -1],
                                             Z_hat=Z_hat,
                                             Z=Z,
                                             G=G,
                                             total_num_fragments=total_num_fragments)

        total_loss = loss_terms['likelihood'] +\
            loss_terms['graph_prior'] +\
            loss_terms['kl_term'] +\
            loss_terms['graph_entropy'] +\
            loss_terms['f_term']

        # total_loss = loss_terms['likelihood'] + loss_terms['f_term']
        
        # print("Likelihood", loss_terms['likelihood'])
        # print("KL", loss_terms['kl_term'])
        # print('f_term', loss_terms['f_term'])
        # print("Total_loss", total_loss)

        return X_lag, Z, X_hat, F, G, total_loss, loss_terms

    def reparameterize(self,
                       mean: torch.Tensor,
                       logvar: torch.Tensor):
        """Reparameterization trick to sample from a normal distribution
        Args:
            mean (torch.Tensor): Mean of the latent variable
            logvar (torch.Tensor): Log variance of the latent variable
        Returns:
            z (torch.Tensor): Sampled latent variable
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps*std + mean

    def forward(self,
                X: torch.Tensor,
                spatial_factors: torch.Tensor=None) -> Tuple[torch.Tensor]:
        """Forward model for Spacy

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, num_variates, timesteps, num_grid_points]
        Returns:
            X_lag (torch.Tensor): Time lagged tensor of shape [n_fragments, num_variates, lag+1, num_grid_points]
            Z (torch.Tensor): Inferred latent timeseries of shape [n_fragments, num_nodes]
            X_hat (torch.Tensor): Reconstructed tensor of shape [n_fragments, num_variates, num_grid_points]
            G (torch.Tensor): Graph of shape [lag+1, num_nodes, num_nodes]
        """
        X_lag = convert_data_to_timelagged(X, self.lag)

        batch, num_variates, timesteps, num_grid_points = X_lag.shape

        # get the inferred mean and logvar
        Z = self.f_tilde(X_lag)
        # Z.shape: (batch, num_variates, lag+1, 2*num_nodes)

        # TODO: fix
        assert Z.shape[1] == 1, "Only one variate supported"

        Z = Z.view(batch, self.lag+1, 2*self.num_nodes)

        # Z.shape: (batch, lag+1, 2*num_nodes)

        Z_mean = Z[..., :self.num_nodes]
        Z_logvar = Z[..., self.num_nodes:]

        # sample Z
        Z = self.reparameterize(Z_mean, Z_logvar)
        
        # sample a graph
        G = self.temporal_graph_dist.sample_graph()

        # pass through the SCM
        Z_hat = self.scm_model(Z, G)

        # sample spatial factor (if no ground truth is provided)
        if spatial_factors != None:
            F = spatial_factors[0]
        else:
            F = self.spatial_factors.get_spatial_factors()

        # decode Z
        X_hat = self.spatial_decoder(Z[:, -1], F)

        # print("Z", Z[0, -1])
        # print("F", F)
        # print("Z_logvar", Z_logvar[0, -1])

        return X_lag, X_hat, Z_mean, Z_logvar, Z_hat, Z, G, F

    def inference(self,
                X: torch.Tensor,
                spatial_factors: torch.Tensor=None) -> Tuple[torch.Tensor]:
        """Forward model for Spacy (inference)

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, num_variates, timesteps, num_grid_points]
        Returns:
            X_lag (torch.Tensor): Time lagged tensor of shape [n_fragments, num_variates, lag+1, num_grid_points]
            Z (torch.Tensor): Inferred latent timeseries of shape [n_fragments, num_nodes]
            X_hat (torch.Tensor): Reconstructed tensor of shape [n_fragments, num_variates, num_grid_points]
            G (torch.Tensor): Graph of shape [lag+1, num_nodes, num_nodes]
        """


        X_lag = convert_data_to_timelagged(X, self.lag)

        batch, num_variates, timesteps, num_grid_points = X_lag.shape

        # get the inferred mean and logvar
        Z = self.f_tilde(X_lag)
        # Z.shape: (batch, num_variates, lag+1, 2*num_nodes)

        # TODO: fix
        assert Z.shape[1] == 1, "Only one variate supported"
        Z = Z.view(batch, self.lag+1, 2*self.num_nodes)
        # Z.shape: (batch, lag+1, 2*num_nodes)

        Z_mean = Z[..., :self.num_nodes]
        Z_logvar = Z[..., self.num_nodes:]

        # sample Z
        Z = self.reparameterize(Z_mean, Z_logvar)
        
        # sample a graph
        G = self.temporal_graph_dist.sample_graph()

        # pass through the SCM
        Z_hat = self.scm_model(Z, G)

        # predicting last timestamp
        Z_pred = self.scm_model.predict(Z,G)

        # sample spatial factor
        #########################################
        if spatial_factors != None:
            F = spatial_factors[0]
        #########################################
        else:
            F = self.spatial_factors.get_spatial_factors()

        # decode Z
        X_hat = self.spatial_decoder(Z[:, -1], F)

        X_pred = self.spatial_decoder(Z[:, -1], F)
        predictions = {'Z':Z_pred, 'X':X_pred}
        # print("Z", Z[0, -1])
        # print("F", F)
        # print("Z_logvar", Z_logvar[0, -1])

        return X_lag, X_hat, Z_mean, Z_logvar, Z_hat, Z, G, F, predictions


    def get_module_dict(self):
        """Get the module dictionary for the model
        Returns:
            dict: Dictionary of modules
        """
        return {
            "scm": self.scm_model,
            "graph": self.temporal_graph_dist,
            "encoder": self.f_tilde,
            "spatial_factors": self.spatial_factors,
            "spatial_decoder": self.spatial_decoder
        }
