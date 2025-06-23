import torch.nn as nn
import torch

class MLPBlock(nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 layer_norm: bool = True,
                 activation: str = 'leaky_relu',
                 skip_connection: bool = True
                 ):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.skip_connection = skip_connection
        
        self.w = nn.Linear(input_dim, out_dim)
        assert not (batch_norm and layer_norm), "Both batch norm and layer norm cannot be present!"
        
        if dropout != 0.0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = nn.Identity()
            
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(f"Activation type {activation} not implemented")
        
        if batch_norm:
            self.norm_layer = nn.BatchNorm1d(input_dim)
        elif layer_norm:
            self.norm_layer = nn.LayerNorm(input_dim)
        else:
            self.norm_layer = nn.Identity()
            
    def forward(self, x):
        z = self.norm_layer(x)
        z = self.w(z)
        z = self.activation(z)
        z = self.dropout_layer(z)
        
        if self.skip_connection and self.input_dim == self.out_dim:
            return z + x
        else:
            return z
        