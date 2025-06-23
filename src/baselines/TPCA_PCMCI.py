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
from src.utils.evaluation import get_permutation_and_eval_mapping, get_permutation_and_graph, pcmci, compute_mcc
from src.baselines.TensorPCA import TensorPCA
from datetime import datetime

import os
from sklearn.metrics import f1_score

# SAVAR imports

def get_savedir(data_string, seed):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    string = f'seed_{seed}_{dt_string}'
    return os.path.join('logs', data_string, string)

class TPCA_PCMCI(pl.LightningModule):

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

        self.data_string = f'tpca_numnodes_{self.model.num_nodes}_size_{self.model.nx}'
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
        # if self.test:
        #     total_num_fragments = 100 * (timesteps - self.model.lag)
        # else:
        #     total_num_fragments = len(
        #         self.trainer.train_dataloader.dataset) * (timesteps - self.model.lag)
        # if (Z_gt == None and G == None) == False:
        tpca = TensorPCA(lag = self.model.lag,
                            num_nodes = self.model.num_nodes,
                            nx = self.model.nx,
                            ny = self.model.ny,
                            num_variates = self.model.num_variates)
        Z_pred = tpca.fit_transform(X)
        dim_dict = {'nx': tpca.nx, 
                'ny': tpca.ny, 
                'num_nodes': tpca.num_nodes,
                'num_variates': tpca.num_variates,
                'lag': tpca.lag}
        G_pred,mcc = get_permutation_and_graph(Z_pred, Z_gt, dim_dict)



        # else:
        #     lag = self.model.lag
        #     num_nodes = self.model.num_nodes
        #     # np.random.rand(3, 10, 10)
        #     dm_method = DmMethod(data=X.squeeze(0).detach().cpu().numpy().T, adj=np.zeros((lag, num_nodes, num_nodes)), 
        #                         mode_weight=None, pc_alpha=1e-3, 
        #                         correct_permutation = False, verbose=False)
        # dm_method.perform_dm()
        # dm_method.get_pcmci_results()
        # dm_method.get_phi_and_predict()

        return X, Z_gt, Z_pred.detach().cpu(), G, G_pred.detach().cpu(), mcc

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
        
        G = []
        if (Z_gt == [] and G == []) == False:
            G = graph[0]
            MCC = 0
        Z_preds = []
        modes = []
        for i in range(batch_size):
            if (Z_gt == [] and G == []) == False:
                _, _, Z_pred, _, G_pred, mcc = self.model_step((X[i],G,Z_gt[i], spatial_factors[i]))
                Z_preds.append(Z_pred)
                MCC += mcc
            else:
                dm_method= self.model_step((X[i],None,None, None))
                Z_temp = dm_method.get_signal()['varimax']
                Z_varimax.append(Z_temp)
                modes.append(dm_method.weights["varimax"])
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
        Z_preds = np.array(Z_preds).squeeze(0)
        # G_latent = G_pred
        G_latent = pcmci(Z_preds, self.model.lag, pc_alpha=1e-2)
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
        if batch_idx%100 == 99:
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