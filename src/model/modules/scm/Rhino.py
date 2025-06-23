from sklearn.covariance import log_likelihood
import torch
import torch.nn as nn

from src.model.modules.scm.RhinoSCM import RhinoSCM
from src.model.modules.scm.likelihood.MSELikelihood import MSELikelihood


class Rhino(nn.Module):

    def __init__(self,
                 decoder: RhinoSCM,
                 likelihood_model: nn.Module) -> None:

        super().__init__()
        self.decoder = decoder
        self.likelihood_model = likelihood_model
        
    def forward(self, X, G):
        return self.decoder(X, G)
    
    def calculate_likelihood(self, X_true, X_pred, X_history=None, expanded_G=None, mean = False):
        return self.likelihood_model.calculate_likelihood(X_true, X_pred, X_history, expanded_G, mean)
    
    def predict(self, X, G):

        # Predict
        X_pred = self.decoder.autoregressive_forward(X,G)

        # Sample noise
        X_history = X[:,:-1,:]
        noise = self.likelihood_model.sample(1, X_history.unsqueeze(-1), G.unsqueeze(0).expand(X.shape[0],-1,-1,-1), None)
        return X_pred + noise.squeeze()