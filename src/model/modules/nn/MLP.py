# This class defines a multi-layer perceptron (MLP) neural network with customizable architecture
# including options for skip connections, batch normalization, layer normalization, and activation
# functions.
import torch.nn as nn
import torch
from src.model.modules.nn.MLPBlock import MLPBlock

class MLP(nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 layer_norm: bool = True,
                 nonlinearity: str = 'leaky_relu',
                 output_activation: str = None,
                 skip_connection: bool = True
                 ):
        super().__init__()
        
        assert num_layers >= 1, "Number of layers need to be >=1"
        
        if num_layers == 1:
            self.net = MLPBlock(input_dim=input_dim,
                                out_dim=out_dim,
                                dropout=dropout,
                                batch_norm=batch_norm,
                                layer_norm=layer_norm,
                                activation=output_activation,
                                skip_connection=skip_connection)
        else:
            module_list = [MLPBlock(input_dim=input_dim,
                                    out_dim=hidden_dim,
                                    dropout=dropout,
                                    batch_norm=False,
                                    layer_norm=False,
                                    activation=nonlinearity,
                                    skip_connection=skip_connection)]
            
            for i in range(num_layers-2):
                module_list.append(MLPBlock(input_dim=hidden_dim,
                                    out_dim=hidden_dim,
                                    dropout=dropout,
                                    batch_norm=batch_norm,
                                    layer_norm=layer_norm,
                                    activation=nonlinearity,
                                    skip_connection=skip_connection))          
            
            module_list.append(MLPBlock(input_dim=hidden_dim,
                                    out_dim=out_dim,
                                    dropout=0.0,
                                    batch_norm=batch_norm,
                                    layer_norm=layer_norm,
                                    activation=output_activation,
                                    skip_connection=False)) 
            self.net = nn.Sequential(*module_list)
            
    def forward(self, x):
        return self.net(x)
    