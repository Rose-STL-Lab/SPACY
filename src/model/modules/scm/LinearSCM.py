import lightning.pytorch as pl
import torch.nn as nn
import torch
import math
import numpy as np


class LinearSCM(pl.LightningModule):

    def __init__(self,
                 lag: int,
                 num_nodes: int
                 ):
        super().__init__()

        self.lag = lag
        self.num_nodes = num_nodes
        self.w = nn.Parameter(
            torch.randn(self.lag+1, self.num_nodes, self.num_nodes, device=self.device)*0.5, requires_grad=True
        )



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
        
        return torch.einsum("lij,bli->bj", (self.w * A).flip([0]), X_input)
    
    def autoregressive_forward(self, X_input: torch.Tensor, A: torch.Tensor):
        """
            Args:
                X_input: input data of shape (batch, lag+1, num_nodes)
                A: adjacency matrix of shape ( (lag+1), num_nodes, num_nodes)
            """

        batch, L, num_nodes = X_input.shape
        lag = L-1
        # ensure we have the correct shape
        # assert (A.shape[0] == lag+1 and A.shape[1] ==
        #         num_nodes and A.shape[2] == num_nodes)
        X_lag = X_input[:,:-1,:]
        A_weighted_lag = (self.w[1:] * A[1:]).flip([0])
        X_curr = torch.einsum("lij,bli->bj", A_weighted_lag, X_lag)
        A0 = A[0]
        topo_order = self.topological_order(A0)

        A_curr = self.w[0]*A0
        for j in topo_order:
            # Get parents of node j (indices where A0[:, j] != 0)
            parents = torch.nonzero(A_curr[:, j], as_tuple=True)[0]
            
            # Compute contribution from parents
            if len(parents) > 0:
                # Weighted sum: sum(A0[i,j] * X_curr[i]) for parents i
                parent_contrib = torch.einsum('i,bi->b', A_curr[parents, j], X_curr[:, parents])
            else:
                parent_contrib = 0.0
            
            # Add external input (e.g., X_curr or noise)
            X_curr[:, j] = parent_contrib + X_curr[:, j]
        return X_curr

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

