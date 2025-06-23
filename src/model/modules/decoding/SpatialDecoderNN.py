import lightning.pytorch as pl
import torch.nn as nn
import torch
import math
import numpy as np
from src.model.modules.nn.MLP import MLP
from lightning.pytorch import seed_everything


class SpatialDecoderNN(nn.Module):

    def __init__(self,
                 nx: int,
                 ny: int,
                 num_variates: int,
                 embedding_dim: int,
                 lag: int,
                 num_nodes: int,
                 num_dense_layers: int = 2,
                 skip_connection: bool = True,
                 scale: float = 1e-1
                 ):

        super().__init__()

        self.nx = nx
        self.ny = ny
        self.num_variates = num_variates
        self.num_nodes = num_nodes

        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter((
            torch.randn(self.num_variates, self.nx*self.ny,
                        self.embedding_dim)*scale
        ), requires_grad=True)

        self.g = MLP(input_dim=self.embedding_dim + 1,
                           out_dim=1,
                           hidden_dim=64,
                           num_layers=2)

    def forward(self, Z, F):
        X_hat = torch.einsum("bd,vdl->bvl", Z, F)
        E = self.embeddings.expand(X_hat.shape[0], -1, -1, -1)
        X_in = torch.cat((X_hat.unsqueeze(-1), E), dim=-1)

        X_hat = self.g(X_in).squeeze(-1)
        # print(self.embeddings.grad)
        return X_hat