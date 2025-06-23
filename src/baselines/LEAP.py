from typing import Tuple

import torch
import torch.nn as nn
from src.utils.leap_utils.components.transition import MBDTransitionPrior
from src.utils.leap_utils.components.mlp import Inference
from src.utils.leap_utils.components.conv import SyntheticConvDecoder, SyntheticConvEncoder
from src.utils.leap_utils.components.tc import Discriminator, permute_dims
from src.utils.leap_utils.components.transforms import ComponentWiseSpline
from src.utils.reshaping import convert_data_to_timelagged
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.distributions as D

import numpy as np
# import matplotlib.pyplot as plt


class LEAP(pl.LightningModule):

    def __init__(self,
                 lag: int,
                 num_nodes: int,
                 nx: int,
                 ny: int,
                 num_variates: int,
                 nc: int,
                 length: int,
                 z_dim: int,
                 z_dim_trans: int,
                 hidden_dim: int,
                 trans_prior: str,
                 infer_mode: str,
                 bound: int,
                 count_bins: int,
                 order: str,
                 lr: float,
                 l1: float,
                 beta: float,
                 gamma: float,
                 sigma: float,
                 bias: bool = False,
                 decoder_dist='gaussian'
                 ):
        # TODO: documentation

        super().__init__()
        self.lag = lag
        self.num_nodes = num_nodes
        self.num_variates = num_variates
        self.nx = nx
        self.ny = ny
        self.use_gt_agg = True

        # Leap parameters
        self.nc = nc
        self.length = length
        self.z_dim = z_dim
        self.z_dim_trans = z_dim_trans
        self.hidden_dim = hidden_dim
        self.trans_prior = trans_prior
        self.infer_mode = infer_mode
        self.bound = bound
        self.count_bins = count_bins
        self.order = order
        self.lr = lr
        self.l1 = l1
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.bias = bias
        self.decoder_dist = decoder_dist

        # Recurrent inference
        if infer_mode == 'R':

            self.enc = SyntheticConvEncoder(z_dim=z_dim, nc=nc, nf=hidden_dim, h=self.ny, w=self.nx)
            for name, param in self.enc.named_parameters():
                print(f'Layer: {name} | Number of parameters: {param.numel()}')
            
        

            self.dec = SyntheticConvDecoder(z_dim=z_dim, nc=nc, nf=hidden_dim, h=self.ny, w=self.nx)
            for name, param in self.dec.named_parameters():
                print(f'Layer: {name} | Number of parameters: {param.numel()}')
            # Bi-directional hidden state rnn
            self.rnn = nn.GRU(input_size=z_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=1, 
                              batch_first=True, 
                              bidirectional=True)
            for name, param in self.rnn.named_parameters():
                print(f'Layer: {name} | Number of parameters: {param.numel()}')
            # Inference net
            self.net = Inference(lag=lag,
                                 z_dim=z_dim, 
                                 hidden_dim=hidden_dim, 
                                 num_layers=1)
            for name, param in self.net.named_parameters():
                print(f'Layer: {name} | Number of parameters: {param.numel()}')
        # elif infer_mode == 'F':
        #     from .leap_utils.components.beta import BetaVAE_CNN, BetaVAE_Physics
        #     # self.net = BetaVAE_CNN(nc=nc, 
        #     #                        z_dim=z_dim,
        #     #                        hidden_dim=hidden_dim)
        #     self.net = BetaVAE_Physics(nc=nc, 
        #                                z_dim=z_dim,
        #                                hidden_dim=hidden_dim)

        # Initialize transition prior
        if trans_prior == 'L':
            self.transition_prior = MBDTransitionPrior(lags=lag, 
                                                       latent_size=self.z_dim_trans, 
                                                       bias=bias)
        
        # Spline flow model to learn the noise distribution
        self.spline = ComponentWiseSpline(input_dim=self.z_dim_trans,
                                          bound=bound,
                                          count_bins=count_bins,
                                          order=order)
        # FactorVAE
        self.discriminator = Discriminator(z_dim = self.z_dim_trans*1)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim_trans))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim_trans))
    
    

    @property
    def base_dist_trans(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean
        
    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        return recon_loss
    
    def inference(self, ft, random_sampling=True):
        ## bidirectional lstm/gru 
        # input: (batch, seq_len, z_dim)
        # output: (batch, seq_len, z_dim)
        output, h_n = self.rnn(ft)
        batch_size, length, _ = output.shape
        # beta, hidden = self.gru(ft, hidden)
        ## sequential sampling & reparametrization
        ## transition: p(zt|z_tau)
        zs, mus, logvars = [], [], []
        for tau in range(self.lag):
            zs.append(torch.ones((batch_size, self.z_dim), device=output.device))

        for t in range(length):
            mid = torch.cat(zs[-self.lag:], dim=1)
            inputs = torch.cat([mid, output[:,t,:]], dim=1)    
            distributions = self.net(inputs)
            mu = distributions[:, :self.z_dim]
            logvar = distributions[:, self.z_dim:]
            zt = self.reparameterize(mu, logvar, random_sampling)
            zs.append(zt)
            mus.append(mu)
            logvars.append(logvar)

        zs = torch.squeeze(torch.stack(zs, dim=1))
        # Strip the first L zero-initialized zt 
        zs = zs[:,self.lag:]
        mus = torch.squeeze(torch.stack(mus, dim=1))
        logvars = torch.squeeze(torch.stack(logvars, dim=1))
        return zs, mus, logvars
    
    def infer_causal_graph(self,
                           X_lag: torch.Tensor,
                           X_hat: torch.Tensor,
                           zs: torch.Tensor,
                           mus: torch.Tensor,
                           logvars: torch.Tensor,
                           mode_weights: torch.Tensor):
        batch_size, nc, length, grid_size = X_lag.shape
        X_lag = X_lag.detach().cpu().numpy()
        X_hat = X_hat.detach().cpu().numpy()
        mus = mus.detach().cpu().numpy()
        logvars = logvars.detach().cpu().numpy()
        zs = zs.detach().cpu().numpy()
        mode_weights = mode_weights.detach().cpu().numpy()

        X_0 = X_lag[0]
        mu_0 = mus[0]
        X_0 = X_0.reshape(self.ny*self.nx, -1)
        X_latent = mode_weights@X_0
        X_latent = X_latent[self.num_variates-1].T
        C = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes): 
            C[i] = -np.abs(np.corrcoef(X_latent, mu_0, rowvar=False)[i,self.num_nodes:])
        
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(C)


        mus = mus[:,:, col_ind]
        from sklearn.feature_selection import mutual_info_regression

        z0 = mus[:,0,:][:,col_ind]
        prediction = []
        prediction.append(np.zeros((self.num_nodes, self.num_nodes)))
        for i in range(1, self.lag+1):

            z_i = mus[:,i,:][:,col_ind]
            
            X = z0
            Y = z_i

            mask = [ ]
            thres = 0.085
            n_neighbors = 10
            for idx in range(self.num_nodes):
                mi = mutual_info_regression(X, Y[:,idx], n_neighbors=n_neighbors)
                mask = mask + list((mi / np.max(mi)) > thres)

            mask = np.array(mask).reshape(self.num_nodes,self.num_nodes)
            for i in range(self.num_nodes):
                mask[i,i] = 1   
            mask = mask.astype(int)
            prediction.append(mask)

        prediction = np.array(prediction)
        G_predict = torch.from_numpy(prediction).cuda()
        return G_predict


    def compute_loss_terms(self,
                           X: torch.Tensor,
                           x_recon: torch.Tensor,
                           mus: torch.Tensor,
                           logvars: torch.Tensor,
                           zs: torch.Tensor):

        batch_size, nc, length, _ = X.shape
        h, w  = self.ny, self.nx
        x_recon = x_recon.view(batch_size, self.num_variates, length, h*w)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        sum_log_abs_det_jacobians = 0
        recon_loss = self.reconstruction_loss(X[:,:,:self.lag], x_recon[:,:,:self.lag], self.decoder_dist)/self.lag + \
                     (self.reconstruction_loss(X[:,:,self.lag:], x_recon[:,:,self.lag:], self.decoder_dist))/(length-self.lag)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        # Future KLD
        log_qz_laplace = log_qz[:,self.lag:, :self.z_dim_trans]
        residuals, logabsdet = self.transition_prior(zs[:,:,:self.z_dim_trans])
        sum_log_abs_det_jacobians += logabsdet
        es, logabsdet = self.spline(residuals.contiguous().view(-1, self.z_dim_trans))
        es = es.reshape(batch_size, length-self.lag, self.z_dim_trans)
        logabsdet = torch.sum(logabsdet.reshape(batch_size,length-self.lag), dim=1)
        sum_log_abs_det_jacobians += logabsdet
        log_pz_laplace = torch.sum(self.base_dist_trans.log_prob(es), dim=1) + sum_log_abs_det_jacobians
        
        # KLD for non-causal transition variables (static content)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (length-self.lag)
        kld_laplace = kld_laplace.mean()

        # L1 penalty to encourage sparcity in causal matrix
        l1_loss = sum(torch.norm(param, 1) for param in self.transition_prior.transition.parameters())

        D_z = self.discriminator(residuals.contiguous().view(batch_size, -1))
        tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.sigma * tc_loss + self.l1 * l1_loss

        # Discriminator training
        residuals = residuals.detach()
        # D_z = self.discriminator(residuals.contiguous().view(batch_size, -1))
        ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        zs_perm = zs
        zs_perm = zs_perm.reshape(batch_size, length, self.z_dim)
        residuals_perm, _ = self.transition_prior(zs_perm[:,:,:self.z_dim_trans])
        residuals_perm = permute_dims(residuals_perm.contiguous().view(batch_size, -1)).detach()
        D_z_pperm = self.discriminator(residuals_perm)
        D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))            
        loss_terms = {
            "loss": loss,
            "recon_loss": recon_loss,
            "kld_normal": kld_normal,
            "kld_laplace": kld_laplace,
            "D_tc_loss": D_tc_loss
        }

        return loss_terms

    def compute_loss(self,
                     X: torch.Tensor,
                     mode_weights: torch.Tensor = None,
                     mapping: torch.Tensor = None):
        """Compute total loss and the loss terms for STCD

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, num_variates, timesteps, nx, ny]
            total_num_fragments (int): The total number of fragments in the training set.
        Returns:
            loss: total loss
            loss_terms: dictionary of loss terms
        """
        X_lag, zs, X_hat, mus, logvars = self(X,
                                     mode_weights=mode_weights,
                                     mapping=mapping)

        loss_terms = self.compute_loss_terms(X=X_lag,
                                             x_recon=X_hat,
                                             mus=mus,
                                             logvars=logvars,
                                             zs=zs)

        total_loss = loss_terms['loss']
        # total_loss = loss_terms['spatial_prior']
        latent = {'zs': zs,
                  'mus': mus,
                  'logvars': logvars}
        return X_lag, latent, X_hat, total_loss, loss_terms

    def forward(self,
                X: torch.Tensor,
                mode_weights: torch.Tensor = None,
                mapping: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """Forward model for LEAP

        Args:
            X (torch.Tensor): Input tensor of shape [batch_size, num_variates, timesteps, num_grid_points]
            mode_weights (torch.Tensor): If not none, model uses the mode_weights for aggregation and deaggregation
            mapping (torch.Tensor): If not none, model uses the mapping for aggregation and deaggregation.
        Returns:
            X_lag (torch.Tensor): Time lagged tensor of shape [n_fragments, num_variates, lag+1, num_grid_points]
            Z (torch.Tensor): Inferred latent timeseries of shape [n_fragments, num_nodes]
            X_hat (torch.Tensor): Reconstructed tensor of shape [n_fragments, num_variates, num_grid_points]
            G (torch.Tensor): Graph of shape [lag+1, num_nodes, num_nodes]
        """

        # assert (mode_weights is None or mapping is None) 
        # "Both mapping and mode weights cannot be input"
        
        X_lag = convert_data_to_timelagged(X, self.lag)
        X = X.permute(2,0,1,3)
        batch_size, nc, length, grid_size = X_lag.shape
        X_flat = X_lag.view(-1, nc, self.ny, self.nx)

        if self.infer_mode == 'R':
            ft = self.enc(X_flat)
            ft = ft.view(batch_size, length, -1)
            zs, mus, logvars = self.inference(ft)
            zs_flat = zs.contiguous().view(-1, self.z_dim)
            X_hat = self.dec(zs_flat)
        
        X_hat = X_hat.view(batch_size, nc, length, grid_size)
        # G_predict = self.infer_causal_graph(X_lag=X_lag,
        #                                     X_hat=X_hat,
        #                                     mus=mus,
        #                                     logvars=logvars,
        #                                     zs=zs,
        #                                     mode_weights=mode_weights)
        
        return X_lag, zs, X_hat, mus, logvars
