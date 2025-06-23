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

class Rhino(pl.LightningModule):

    def __init__(self,
                 lag: int,
                 num_nodes: int,
                 scm_model: nn.Module,

                 # loss terms weightage,
                 graph_sparsity_factor: float,
                 ):
        # TODO: documentation

        super().__init__()
        self.lag = lag
        self.num_nodes = num_nodes

        self.scm_model = scm_model
        self.temporal_graph_dist = TemporalAdjacencyMatrix(input_dim=self.num_nodes,
                                                           lag=self.lag)


        self.graph_sparsity_factor = graph_sparsity_factor


    def compute_loss_terms(self,
                           Z_lag: torch.Tensor,
                           Z_hat: torch.Tensor,
                           G: torch.Tensor,
                           total_num_fragments: int):

        batch = Z_lag.shape[0]
        expanded_G = G.unsqueeze(0).repeat(batch, 1, 1, 1)

        # calculate graph loss terms
        graph_sparsity = self.temporal_graph_dist.calculate_sparsity(G)
        dagness_penalty = self.temporal_graph_dist.calculate_dagness_penalty(
            G[0])/total_num_fragments

        graph_sparsity_term = self.graph_sparsity_factor * graph_sparsity
        graph_prior_term = graph_sparsity_term / total_num_fragments
        graph_entropy = -self.temporal_graph_dist.entropy() / total_num_fragments

        # cd_loss = torch.sum(torch.square(Z-Z_hat))/batch
        cd_loss = self.scm_model.calculate_likelihood(X_true=Z_lag.squeeze(1)[:,-1,:], X_pred=Z_hat,
                                                      X_history=Z_lag.squeeze(1)[:,:-1,:], expanded_G = expanded_G)
        # D = Z_logvar.shape[-1]
        # z_entropy = -0.5*(torch.sum(Z_logvar) + D *
        #                   (1 + math.log(2*math.pi))) / (batch*self.num_nodes)
        likelihood_term = cd_loss
        # print(f'cd_loss: {cd_loss}')
        
        loss_terms = {
            # graph prior terms
            'graph_sparsity': graph_sparsity_term,
            'dagness_penalty': dagness_penalty,
            'graph_entropy': graph_entropy,
            'graph_prior': graph_prior_term,
            'likelihood': likelihood_term,
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

        Z_lag, Z_hat, G = self(X)

        loss_terms = self.compute_loss_terms(Z_lag=Z_lag,
                                             Z_hat=Z_hat,
                                             G=G,
                                             total_num_fragments=total_num_fragments)

        total_loss = loss_terms['likelihood'] +\
            loss_terms['graph_prior'] +\
            loss_terms['graph_entropy']
        
        # print("Likelihood", loss_terms['likelihood'])
        # print("KL", loss_terms['kl_term'])
        # print("Total_loss", total_loss)

        return Z_lag, Z_hat, G, total_loss, loss_terms

    def forward(self,
                Z: torch.Tensor) -> Tuple[torch.Tensor]:
        
        Z_lag = convert_data_to_timelagged(Z.unsqueeze(1), self.lag)
        # Z_lag = convert_data_to_timelagged(Z, self.lag)


        # batch = Z_lag.shape[0]
        # TODO: fix
        
        # sample a graph
        G = self.temporal_graph_dist.sample_graph()

        # pass through the SCM
        Z_hat = self.scm_model(Z_lag.squeeze(1), G)

        return Z_lag, Z_hat, G

    def get_module_dict(self):
        return {
            "scm": self.scm_model,
            "graph": self.temporal_graph_dist,
        }