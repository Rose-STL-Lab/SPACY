import torch
import torch.nn as nn


class MSELikelihood(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def calculate_likelihood(self, X_true, X_pred, X_history = None, expanded_G = None, mean = False):
        return self.mse(X_pred, X_true)
