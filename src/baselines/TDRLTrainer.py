import lightning.pytorch as pl
import torch
from typing import Tuple, Dict, Any

# metrics
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import AUROC
from torchmetrics.aggregation import MeanMetric, MinMetric
from src.utils.data_normalization import standardize
import torch.nn as nn
from src.utils.evaluation import get_permutation_and_eval_mapping, get_permutation_and_graph, pcmci, compute_mcc
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore")

class TDRLTrainer(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        compile: bool = False
    ) -> None:
        """Initialize the time-series causal discovery model.

        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model
        """

        super().__init__()

        self.save_hyperparameters()

        # metric objects for calculating and averaging accuracy across batches
        self.val_f1 = F1Score(task='binary')
        self.test_f1 = F1Score(task='binary')

        self.val_roc = AUROC(task='binary')
        self.test_roc = AUROC(task='binary')
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.model = model
        self.use_gt_agg = self.model.use_gt_agg
        self.idx = 0
        self.rhino = False
        # self.scm_model = scm_model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param X: A tensor of time series data, of shape (batch, num_samples, timesteps, num_nodes).
        :return: A tensor of logits.
        """
        return self.model(X)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_f1.reset()
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
        mode_weights = None
        G = G[0]
        # W = W[0]
        if self.use_gt_agg:
            # mode_weights = mode_weights[0].view(
            #     self.model.num_variates, self.model.num_nodes, -1)
            X_lag, latent, X_hat, total_loss, loss_terms = self.model.compute_loss(
                X, mode_weights)
        else:
            X_lag, latent, X_hat, total_loss, loss_terms = self.model.compute_loss(
                X)
            
        Z_pred = latent
        X_permute = {'X_lag': X_lag, 'X_hat': X_hat, 'mode_weights': mode_weights}
        return total_loss, loss_terms, G, X_permute, Z_gt, Z_pred

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_terms, G, X_permute, Z_gt, Z_pred = self.model_step(
            batch)
        # update and log metrics
        self.train_loss(loss)

        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_terms, G, X_permute, Z_gt, Z_pred = self.model_step(
            batch)
        
        dim_dict = {'nx': self.model.nx, 
                    'ny': self.model.ny, 
                    'num_nodes': self.model.num_nodes,
                    'num_variates': self.model.num_variates,
                    'lag': self.model.lag}
        # permute the nodes of G_pred based on W and W_pred, and obtain clustering accuracy
        # P, cluster_acc = get_permutation_and_eval_mapping(W, W_pred)
        # P = P.to(self.device)
        # G_pred = G_pred[:, P][:, :, P]

        G_pred,mcc = get_permutation_and_graph(Z_pred, Z_gt, dim_dict)
        G_pred = G_pred.to(self.device)
        ###################################
        # save pictures
        import matplotlib.pyplot as plt
        # if not self.use_gt_agg:
        #     W_mask = torch.argmax(W_pred, dim=1).reshape(-1,
        #                                                  self.model.ny, self.model.nx)
        # else:
        #     W_mask = W_pred.reshape(-1, self.model.ny, self.model.nx)
        # for i in range(W_pred.shape[0]):
        #     plt.imsave(f'{i}.png', W_mask[i].detach().cpu().numpy())
        X_true = X_permute['X_lag'].reshape(-1, self.model.ny * self.model.nx)
        X_hat = X_permute['X_hat'].reshape(-1, self.model.ny * self.model.nx)
        if self.idx == 100:
            self.idx = 0
        if self.idx == 0:
            plt.figure()
            plt.imsave(f'X_true.png', X_true[0].detach().cpu().numpy().reshape(self.model.ny, self.model.nx))
            plt.close()
            plt.figure()
            plt.imsave(f'X_hat.png', X_hat[0].detach().cpu().numpy().reshape(self.model.ny, self.model.nx))
            plt.close()
        self.idx += 1
        #################################
        # update and log metrics
        self.val_loss(loss)
        self.val_f1(G_pred, G)
        self.val_roc(G_pred, G)

        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_roc, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/mcc", mcc,
                 on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_loss = self.val_loss.compute()  # get current val acc
        self.val_loss(val_loss)  # update best so far val acc
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(),
                 sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_terms, G, X_permute, Z_gt, Z_pred = self.model_step(
            batch)
        # permute the nodes of G_pred based on W and W_pred, and obtain clustering accuracy
        dim_dict = {'nx': self.model.nx, 
                    'ny': self.model.ny, 
                    'num_nodes': self.model.num_nodes,
                    'num_variates': self.model.num_variates,
                    'lag': self.model.lag}
        # permute the nodes of G_pred based on W and W_pred, and obtain clustering accuracy
        # P, cluster_acc = get_permutation_and_eval_mapping(W, W_pred)
        # P = P.to(self.device)
        # G_pred = G_pred[:, P][:, :, P]

        if not self.rhino:
            G_pred,mcc = get_permutation_and_graph(Z_pred, Z_gt, dim_dict)
            G_pred = G_pred.to(self.device)
        # else:
        #     scm_model = self.scm_model
            


        print("G:", G)
        print("G_pred", G_pred)
        # update and log metrics
        self.test_loss(loss)
        self.test_f1(G_pred, G)
        self.test_roc(G_pred, G)

        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/auroc", self.test_roc, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/mcc", mcc,
                 on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Compiles model if needed

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers 

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # optimizer = self.hparams.optimizer(
        #     params=self.model.parameters())
        modules = {
            "leap": self.model
        }

        parameter_list = [
            {
                "params": module.parameters(),
                "lr": self.model.lr,
                "name": name,
            }
            for name, module in modules.items() if module is not None
        ]

        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        # else:
        return torch.optim.Adam(parameter_list)
