import lightning as pl
from omegaconf import OmegaConf
import torch
from typing import Tuple, Dict, Any

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




def get_savedir(data_string):
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # string = f'{data_string}/{dt_string}'
    return os.path.join('logs', data_string, dt_string)


class RhinoTrainer(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        training_config: AuglagLRConfig,
        compile: bool = False
    ) -> None:
        """Initialize the time-series causal discovery model.

        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model
        """

        super().__init__()

        self.save_hyperparameters()

        # loss function
        self.bce_logits = torch.nn.BCEWithLogitsLoss()
        self.bce = torch.nn.BCELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.val_f1 = F1Score(task='binary')
        self.val_f1_lag = F1Score(task='binary')
        self.val_f1_inst = F1Score(task='binary')

        self.test_f1 = F1Score(task='binary')
        self.test_f1_lag = F1Score(task='binary')
        self.test_f1_inst = F1Score(task='binary')

        self.pcmci_f1 = F1Score(task='binary')
        self.pcmci_f1_lag = F1Score(task='binary')
        self.pcmci_f1_inst = F1Score(task='binary')

        self.val_roc = AUROC(task='binary')
        self.test_roc = AUROC(task='binary')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.model = model
        self.training_config = training_config
        self.lr_scheduler = AuglagLR(config=self.training_config)
        self.loss_calc = AuglagLossCalculator(init_alpha=self.training_config.init_alpha,
                                              init_rho=self.training_config.init_rho)
  
        ##############################################
        # Visualization Param:
        self.validating = True
        self.test = False
        self.idx = 0
        ##############################################

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
        Z, G, Z_gt, spatial_factors = batch
        # timesteps = Z.shape[2]
        timesteps = Z_gt.shape[1]
        # Z_gt = Z_gt.permute(0,2,1)

        ##################################################
        if self.test:
            total_num_fragments = 100 * (timesteps - self.model.lag)
        else:
            total_num_fragments = len(
                self.trainer.train_dataloader.dataset) * (timesteps - self.model.lag)
        ##################################################

        # total_num_fragments = len(
        #     self.trainer.train_dataloader.dataset) * (timesteps - self.model.lag)
        
        G = G[0]
        # W = W[0]

        Z_lag, Z_pred, G_pred, total_loss, loss_terms = self.model.compute_loss(
            Z,
            total_num_fragments=total_num_fragments,
            spatial_factors=None)
        return total_loss, loss_terms, G, G_pred, Z_gt, Z_lag, Z_pred

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
        ####################################################################################
        self.validating = False
        ####################################################################################

        loss, loss_terms, G, G_pred,Z_gt, Z_lag, Z_pred = self.model_step(
            batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        loss = self.loss_calc(loss, loss_terms['dagness_penalty'])
        loss_terms['loss'] = loss

        self.visualize = True
        return loss_terms

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_terms, G, G_pred, Z_gt, Z_lag, Z_pred = self.model_step(
            batch)

        # # permute the nodes of G_pred based on W and W_pred, and obtain clustering accuracy
        # # TODO: Fix assignment
        # W_pred = torch.argmax(F_pred, axis=1)

        # P, cluster_acc = get_permutation_and_eval_mapping(W, W_pred)

        # permute the node of G_pred based on Latent Timeseries
        
        mcc, P = compute_mcc(Z_gt.squeeze(0)[self.model.lag:, :].detach().cpu().numpy().T, Z_lag.squeeze(1)[:,-1,:].detach().cpu().numpy().T, "Pearson")
        P = P.to(self.device)
        G_pred = G_pred[:, P][:, :, P]

        G_probs = self.model.temporal_graph_dist.get_adj_matrix()
        print("G", G)
        print("G_pred", G_probs[:, P][:, :, P])

        # update and log metrics
        self.val_loss(loss)
        self.val_f1(G_pred, G)
        self.val_f1_lag(G_pred[1:], G[1:])
        self.val_f1_inst(G_pred[0], G[0])
        self.val_roc(G_probs, G)
        print(f'Sample_{batch_idx}: ',f1_score(G_pred.detach().cpu().numpy().flatten(), G.detach().cpu().numpy().flatten()))
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1_lag", self.val_f1_lag, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1_inst", self.val_f1_inst, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_roc, on_step=False,
                 on_epoch=True, prog_bar=True)

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

        # TODO: FIX THIS
        self.test = True
        loss, loss_terms, G, G_pred, Z_gt, Z_lag, Z_pred = self.model_step(
            batch)
        # permute the nodes of G_pred based on W and W_pred, and obtain clustering accuracy
        # P, cluster_acc = get_permutation_and_eval_mapping(W, F_pred)

        # mcc, P = compute_mcc(Z_gt.squeeze(0)[self.model.lag:, :].detach().cpu().numpy().T, Z_lag.squeeze(1)[:,-1,:].detach().cpu().numpy().T, "Pearson")
        # P = P.to(self.device)
        # G_pred = G_pred[:, P][:, :, P]

        G_probs = self.model.temporal_graph_dist.get_adj_matrix()
        # print("G", G)
        # print("G_pred", G_probs[:, P][:, :, P])

        # update and log metrics
        self.test_loss(loss)
        self.test_f1(G_pred, G)
        self.test_f1_lag(G_pred[1:], G[1:])
        self.test_f1_inst(G_pred[0], G[0])
        self.test_roc(G_probs, G)


        
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/f1_lag", self.test_f1_lag, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/f1_inst", self.test_f1_inst, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/auroc", self.test_roc, on_step=False,
                 on_epoch=True, prog_bar=True)

        PCMCI = True
        if PCMCI:
            num_samples = Z_gt.shape[0]
            G_pcmci = pcmci(Z_lag.squeeze(1)[:,0].reshape(num_samples,-1,self.model.num_nodes).detach().cpu().numpy(), self.model.lag, pc_alpha = 1e-3)
            G_pcmci = torch.tensor(G_pcmci).to(self.device)
            self.pcmci_f1(G_pcmci,G)
            self.pcmci_f1_lag(G_pcmci[1:],G[1:])
            self.pcmci_f1_inst(G_pcmci[0],G[0])
            self.log("test/pcmci_f1", self.pcmci_f1, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/pcmci_f1_lag", self.pcmci_f1_lag, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/pcmci_f1_inst", self.pcmci_f1_inst, on_step=False,
                    on_epoch=True, prog_bar=True)

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
        modules = self.model.get_module_dict()

        parameter_list = [
            {
                "params": module.parameters(),
                "lr": self.training_config.lr_init_dict[name],
                "name": name,
            }
            for name, module in modules.items() if module is not None
        ]

        # Check that all modules are added to the parameter list
        check_modules = set(modules.values())
        for module in self.parameters(recurse=False):
            assert module in check_modules, f"Module {module} not in module list"

        return torch.optim.Adam(parameter_list)

    def on_after_backward(self) -> None:
        for p in self.parameters():
            if p.grad != None:
                p.grad = torch.nan_to_num(p.grad)
        # print("GRAD", self.model.aggregator.V.grad.shape)
        # for i in range(self.model.num_nodes):
            # print(f"grad {i}", self.model.aggregator.V.grad[0, i])
        return super().on_after_backward()

    def configure_callbacks(self):
        """Create a callback for the auglag callback."""
        return [AuglagLRCallback(self.lr_scheduler, log_auglag=True)]
