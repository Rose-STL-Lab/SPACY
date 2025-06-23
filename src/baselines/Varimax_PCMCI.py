import lightning as pl
from omegaconf import OmegaConf
import torch
from typing import Tuple, Dict, Any
import numpy as np
# metrics
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import AUROC
from torchmetrics.aggregation import MeanMetric, MinMetric
from src.utils.data_normalization import standardize
import torch.nn as nn
from src.utils.evaluation import get_permutation_and_eval_mapping, compute_mcc, pcmci
from src.auglag.AuglagLRCallback import AuglagLRCallback
from src.auglag.AuglagLRConfig import AuglagLRConfig
from src.auglag.AuglagLossCalculator import AuglagLossCalculator
from src.auglag.AuglagLR import AuglagLR
from datetime import datetime

import os
from sklearn.metrics import f1_score

# SAVAR imports
from src.utils.savar_utils.eval_tools import Evaluation, DmMethod
from src.utils.savar_utils.savar import SAVAR
import matplotlib.pyplot as plt
from src.utils.savar_utils.functions import create_random_mode, check_stability
from src.utils.savar_utils.dim_methods import get_varimax_loadings_standard as varimax

def get_savedir(data_string, seed):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    string = f'seed_{seed}_{dt_string}'
    return os.path.join('logs', data_string, string)

class Varimax_PCMCI(pl.LightningModule):

    def __init__(
        self,
        model,
        compile: bool = False
    ) -> None:
        """Initialize the time-series causal discovery model.

        :param compile: Whether to compile the model
        """
        super().__init__()

        self.save_hyperparameters()

        # loss function (might still be used for evaluation)
        self.bce_logits = torch.nn.BCEWithLogitsLoss()
        self.bce = torch.nn.BCELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.val_f1 = F1Score(task='binary')
        self.val_f1_lag = F1Score(task='binary')
        self.val_f1_inst = F1Score(task='binary')

        self.test_f1 = F1Score(task='binary')
        self.test_f1_lag = F1Score(task='binary')
        self.test_f1_inst = F1Score(task='binary')

        self.val_roc = AUROC(task='binary')
        self.test_roc = AUROC(task='binary')

        # for averaging loss across batches
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.model = model
        # Visualization Param:
        self.validating = True
        self.test = False
        self.idx = 0

        self.data_string = f'varimax_numnodes_{self.model.num_nodes}_size_{self.model.nx}'
        self.seed = torch.get_rng_state()[0].item()
        self.save_dir = get_savedir(self.data_string, self.seed)
        os.makedirs(self.save_dir)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param X: A tensor of time series data, of shape (batch, num_samples, timesteps, num_nodes).
        :return: A tensor of logits.
        """
        return self.model(X)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_f1.reset()
        self.val_f1_inst.reset()
        self.val_f1_lag.reset()
        self.val_loss_best.reset()
        self.val_roc.reset()

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of loss.
            - A dictionary containing the different loss terms.
            - A tensor of ground truth causal graph
            - A tensor of causal graph predictions.
            - A tensor of target spatial mapping.
            - A tensor of predicted spatial mapping.
        """
        X, G, Z_gt, spatial_factors = batch
        variate,timesteps,grid_size = X.shape
        lag = self.model.lag
        num_nodes = self.model.num_nodes
        # if self.test:
        #     total_num_fragments = 100 * (timesteps - self.model.lag)
        # else:
        #     total_num_fragments = len(
        #         self.trainer.train_dataloader.dataset) * (timesteps - self.model.lag)
        Z_varimax = []
        modes = []
        for v in range(variate):
            if (spatial_factors == None) == False:
                mask = (spatial_factors[v] != 0).any(dim=1)
                mode_weights = spatial_factors[v,mask]
                dm_method = DmMethod(data=X[v].detach().cpu().numpy().T, adj=np.zeros((lag, num_nodes//variate, num_nodes//variate)), 
                                    mode_weight=mode_weights.detach().cpu().numpy(), pc_alpha=1e-3, 
                                    correct_permutation = True, verbose=False)
                dm_method.perform_dm()
                Z_temp = dm_method.get_signal()['varimax']
                Z_varimax.append(Z_temp)
                modes.append(dm_method.weights["varimax"])
            else:
                
                # np.random.rand(3, 10, 10)
                dm_method = DmMethod(data=X[v].detach().cpu().numpy().T, adj=np.zeros((lag, num_nodes//variate, num_nodes//variate)), 
                                    mode_weight=None, pc_alpha=1e-3, 
                                    correct_permutation = False, verbose=False)
                dm_method.perform_dm()
                Z_temp = dm_method.get_signal()['varimax']
                Z_varimax.append(Z_temp)
                modes.append(dm_method.weights["varimax"])
        Z_pred = np.vstack(Z_varimax)
        mode_pred = np.vstack(modes)
        breakpoint()
        mcc = 0
        if (spatial_factors == None) == False:
            mcc, p = compute_mcc(Z_pred, Z_gt.detach().cpu().numpy().T, correlation_fn = 'Pearson')
            Z_pred = Z_pred[p]
        # dm_method.get_pcmci_results()
        # dm_method.get_phi_and_predict()
        
        return Z_pred, mode_pred, mcc

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        X, G, Z_gt, spatial_factors = batch
        batch_size,variate,timesteps,grid_size = X.shape
        for i in range(batch_size):
            dm_method, X, G, Z_gt, spatial_factors= self.model_step(
                (X[i],G,Z_gt[i],spatial_factors[i]))
            evaluator = Evaluation(dm_method, Z=Z_gt)
            G_pred = torch.tensor(evaluator.dm_cg)
            # G_probs = self.model.temporal_graph_dist.get_adj_matrix()
            print("G", G)
            # print("G_pred", G_probs)
            loss = evaluator.metrics['mse']
            # update and log metrics
            self.val_loss(loss)
            self.val_f1(G_pred, G)
            self.val_f1_lag(G_pred[1:], G[1:])
            self.val_f1_inst(G_pred[0], G[0])
        # self.val_roc(G_probs, G)
        # print(f'Sample_{batch_idx}: ', f1_score(G_pred.detach().cpu().numpy().flatten(), G.detach().cpu().numpy().flatten()))

        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1_lag", self.val_f1_lag, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1_inst", self.val_f1_inst, on_step=False,
                 on_epoch=True, prog_bar=True)
        # self.log("val/auroc", self.val_roc, on_step=False,
        #          on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        val_loss = self.val_loss.compute()  # get current val acc
        self.val_loss(val_loss)  # update best so far val acc
        self.log("val/loss_best", self.val_loss_best.compute(),
                 sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        X, graph, Z_gt, spatial_factors = batch
        batch_size,variate,timesteps,grid_size = X.shape
        # if variate > 1:
        #     X = torch.einsum('bvtn->btvn', X).reshape(batch_size,1, timesteps, grid_size * variate)  
        #     if (Z_gt == []) == False:
        #         spatial_factors = torch.einsum('bvtn->btvn', spatial_factors).reshape(batch_size,1, -1, grid_size * variate) 
        G = []
        if (Z_gt == [] and G == []) == False:
            G = graph[0]
            MCC = 0
        Z_varimax = []
        modes = []
        for i in range(batch_size):
            if (Z_gt == []) == False:
                Z_pred, mode_pred, mcc = self.model_step((X[i],G,Z_gt[i], spatial_factors[i]))
                Z_varimax.append(Z_pred)
                modes.append(mode_pred)
                MCC += mcc
            else:
                Z_pred, mode_pred, mcc = self.model_step((X[i],None,None, None))
                Z_varimax.append(Z_pred)
                modes.append(mode_pred)
                # evaluator = Evaluation(dm_method, Z=Z_gt[i].detach().cpu().numpy())
            

            # G_pred_corr = torch.tensor(evaluator.dm_cg['varimax_corr'], device = self.device)
            # G_pred_pcmci = torch.tensor(evaluator.dm_cg['varimax_pcmci'], device = self.device)
            # # G_probs = self.model.temporal_graph_dist.get_adj_matrix()
            # print("G", G)
            # print("G_pred_corr", G_pred_corr)
            # print("G_pred_pcmci", G_pred_pcmci)
            # evaluator.obtain_score_metrics(perform_grid = False)
            # MCC += evaluator.metrics['varimax_corr']['mcc']
            # loss = evaluator.metrics['varimax_corr']['mse']

            # G_latent = pcmci(Z_gt[i].detach().cpu().numpy(), self.model.lag, pc_alpha=1e-2)
            # G_latent = torch.tensor(G_latent, device = self.device)
            # self.test_f1(G_latent, G)
            # self.test_f1_lag(G_latent[1:], G[1:])
            # self.test_f1_inst(G_latent[0], G[0])
        Z_v = np.array(Z_varimax).transpose(0,2,1)
        G_latent = pcmci(Z_v, self.model.lag, pc_alpha=1e-3)
        G_latent = torch.tensor(G_latent, device=self.device)
        # self.test_loss(loss)
        if (Z_gt == [] and G == []) == False:
            self.test_f1(G_latent, G)
            self.test_f1_lag(G_latent[1:], G[1:])
            self.test_f1_inst(G_latent[0], G[0])
            # self.test_roc(G_probs, G)
            MCC /= batch_size
        # self.log("test/loss", self.test_loss, on_step=False,
        #         on_epoch=True, prog_bar=True)
            self.log("test/f1", self.test_f1, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/f1_lag", self.test_f1_lag, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/f1_inst", self.test_f1_inst, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/mcc", MCC, on_step=False,
                    on_epoch=True, prog_bar=True)
        if batch_idx%100 == 1:
            torch.save(torch.tensor(np.array(modes)), os.path.join(self.save_dir, f'modes.pt'))
            torch.save(G_latent.detach().cpu(), os.path.join(self.save_dir, f'G_pred.pt'))

    def setup(self, stage: str) -> None:
        """Compiles model if needed

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """No optimizers needed as training is not required."""
        return {}

    def on_after_backward(self) -> None:
        """Adjust gradients to handle NaN values."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad)
        return super().on_after_backward()

    def configure_callbacks(self):
        """No additional callbacks needed as training is not required."""
        return []