import lightning.pytorch as pl
import torch.nn as nn
import torch
import math
import numpy as np
from src.model.modules.nn.MLP import MLP
from lightning.pytorch import seed_everything

class RhinoSCM(pl.LightningModule):

    def __init__(self,
                 embedding_dim: int,
                 lag: int,
                 num_nodes: int,
                 num_dense_layers: int = 2,
                 skip_connection: bool = True
                 ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.lag = lag
        self.num_nodes = num_nodes

        self.embeddings = nn.Parameter((
            torch.randn(self.lag + 1, self.num_nodes,
                        self.embedding_dim, device=self.device) * 0.01
        ), requires_grad=True)  # shape (lag+1, num_nodes, embedding_dim)

        input_dim = 2*self.embedding_dim

        self.f = MLP(input_dim=input_dim,
                     out_dim=1,
                     hidden_dim=self.embedding_dim,
                     num_layers=num_dense_layers,
                     skip_connection=skip_connection)

        self.g = MLP(input_dim=self.embedding_dim+1,
                     out_dim=self.embedding_dim,
                     hidden_dim=self.embedding_dim,
                     num_layers=num_dense_layers,
                     skip_connection=skip_connection)
        # data_dim = 1
        # # seed_everything(0)
        # self.f = nn.Sequential(
        #     nn.Linear(input_dim, self.embedding_dim),
        #     nn.LayerNorm(self.embedding_dim),
        #     nn.LeakyReLU(),
            
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.LayerNorm(self.embedding_dim),
        #     nn.LeakyReLU(),
            
        #     nn.Linear(self.embedding_dim, data_dim),
        # )
        
        # input_dim = 2*self.embedding_dim
        # self.g = nn.Sequential(
        #     nn.Linear(self.embedding_dim+data_dim, self.embedding_dim),
        #     nn.LayerNorm(self.embedding_dim),
        #     nn.LeakyReLU(),
            
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.LayerNorm(self.embedding_dim),
        #     nn.LeakyReLU(),
            
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        # )
    def forward(self, X_input: torch.Tensor, A: torch.Tensor):
        """
        Args:
            X_input: input data of shape (batch, lag+1, num_nodes)
            A: adjacency matrix of shape ( (lag+1), num_nodes, num_nodes)
        """
        batch, L, num_nodes = X_input.shape
        lag = L-1
        # ensure we have the correct shape

        assert (A.shape[0] == lag+1 and A.shape[1] ==
                num_nodes and A.shape[2] == num_nodes)

        E = self.embeddings.expand(
            X_input.shape[0], -1, -1, -1
        )

        X_in = torch.cat((X_input.unsqueeze(-1), E), dim=-1)

        # X_in: (batch, lag+1, num_nodes, embedding_dim+1)
        X_enc = self.g(X_in)

        A_temp = A.flip([0])

        X_sum = torch.einsum("lij,blio->bjo", A_temp, X_enc)
        # (batch, num_nodes, embedding_dim)

        X_sum = torch.cat([X_sum, E[:, 0, :, :]], dim=-1)
        # pass through f network to get the predictions
        return self.f(X_sum).squeeze(-1)  # (batch, num_nodes)
    
    def autoregressive_forward(self, X_input: torch.Tensor, A: torch.Tensor):
        """
        Args:
            X_input: input data of shape (batch, lag+1, num_nodes)
            A: adjacency matrix of shape ( (lag+1), num_nodes, num_nodes)
        """
        batch, L, num_nodes = X_input.shape
        lag = L - 1

        # 1. Process historical time steps (lag) with embeddings and self.g
        X_lag = X_input[:, :-1, :]  # (batch, lag, num_nodes)
        E = self.embeddings.expand(batch, -1, -1, -1)  # (batch, lag+1, num_nodes, embedding_dim)
        E_lag = E[:, :-1, :, :]  # (batch, lag, num_nodes, embedding_dim)

        # Concatenate input with embeddings and encode
        X_in_lag = torch.cat((X_lag.unsqueeze(-1), E_lag), dim=-1)  # (batch, lag, num_nodes, embedding_dim+1)
        X_enc_lag = self.g(X_in_lag)  # (batch, lag, num_nodes, embedding_dim)

        # 2. Aggregate historical information using adjacency
        A_lag = (A[1:]).flip([0])  # (lag, num_nodes, num_nodes)
        X_curr = torch.einsum("lij,blio->bjo", A_lag, X_enc_lag)  # (batch, num_nodes, embedding_dim)

        # 3. Process current time step with autoregressive dependencies
        A0 = A[0]
        A_curr = A0
        topo_order = self.topological_order(A0)

        # Loop over nodes in topological order to respect dependencies
        for j in topo_order:
            parents = torch.nonzero(A_curr[:, j], as_tuple=True)[0]
            if len(parents) > 0:
                # Extract parent features and weights
                parent_features = X_curr[:, parents, :]  # (batch, num_parents, embedding_dim)
                parent_weights = A_curr[parents, j]  # (num_parents,)
                
                # Weight and aggregate parent features
                weighted_parents = parent_weights[None, :, None] * parent_features  # (batch, num_parents, embedding_dim)
                aggregated = torch.sum(weighted_parents, dim=1)  # (batch, embedding_dim)
                
                # Non-linear transformation of aggregated parent contributions
                X_curr[:, j, :] += aggregated  # Update node j's features

        # 4. Combine with embeddings and pass through final network
        E_curr = E[:, 0, :, :]  # (batch, num_nodes, embedding_dim)
        X_sum = torch.cat([X_curr, E_curr], dim=-1)  # (batch, num_nodes, 2 * embedding_dim)
        return self.f(X_sum).squeeze(-1)  # (batch, num_nodes)
    
    def topological_order(self, A0: torch.Tensor) -> list:
        """
        Args:
            A0: Binary adjacency matrix of shape (num_nodes, num_nodes).
        Returns:
            List of node indices in topological order.
        """
        num_nodes = A0.shape[0]
        in_degree = A0.sum(dim=0)  # In-degree for each node
        queue = torch.where(in_degree == 0)[0].tolist()
        topo_order = []

        while queue:
            u = queue.pop(0)
            topo_order.append(u.item() if isinstance(u, torch.Tensor) else u)
            # Get all neighbors (children) of u
            for v in torch.nonzero(A0[u], as_tuple=True)[0]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        if len(topo_order) > num_nodes:
            raise ValueError("A[0] has cycles. Topological sort requires a DAG.")
        return topo_order