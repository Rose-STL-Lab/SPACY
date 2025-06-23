import lightning as pl
from omegaconf import OmegaConf
import torch
from typing import Tuple, Dict, Any

import torchmetrics
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import AUROC
from torchmetrics.aggregation import MeanMetric, MinMetric
from torchmetrics import MeanSquaredError
from src.utils.data_normalization import standardize
import torch.nn as nn
from src.utils.evaluation import get_permutation_and_eval_mapping, compute_mcc, match_latent_indices
from src.auglag.AuglagLRCallback import AuglagLRCallback
from src.auglag.AuglagLRConfig import AuglagLRConfig
from src.auglag.AuglagLossCalculator import AuglagLossCalculator
from src.auglag.AuglagLR import AuglagLR
from datetime import datetime
import os


def get_savedir(data_string, seed):
    """ Generate a directory name for saving logs and results.
    
    Args:
        data_string (str): A string representing the data configuration.
        seed (int): The random seed used for training.
    Returns:
        str: The generated directory name.
    """
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    string = f'seed_{seed}_{dt_string}'
    return os.path.join('logs', data_string, string)


class ModelTrainer(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        training_config: AuglagLRConfig,
        disable_auglag_epoch: int = -1,
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

        self.val_roc = AUROC(task='binary')
        self.test_roc = AUROC(task='binary')

        self.val_f_mse = MeanSquaredError()
        self.test_f_mse = MeanSquaredError()

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
        
        # generate temporary directory for saving visual results
        self.data_string = f'numnodes_{self.model.num_nodes}_size_{self.model.nx}_numvariates_{self.model.num_variates}'
        if self.model.nx == 192 and self.model.ny == 145:
            self.data_string = f'{self.model.data_model}_{self.data_string}'

        self.seed = torch.get_rng_state()[0].item()
        self.save_dir = get_savedir(self.data_string, self.seed)
        os.makedirs(self.save_dir)
        self.disable_auglag_epoch = disable_auglag_epoch

        ##############################################
        # Visualization Param:
        self.validating = True
        self.test = False
        self.idx = 0
        self.F_evolve = []
        self.rho_logvar = []
        ##############################################

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            X: Input tensor of shape (batch, num_variates, lag, num_grid_points).
        Returns:
            torch.Tensor: Output tensor of shape (batch, num_variates, lag, num_grid_points).
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
        self.val_f_mse.reset()

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
        timesteps = X.shape[2]

        # check in training or test
        if self.test:
            total_num_fragments = len(self.trainer.test_dataloaders.dataset) * (timesteps - self.model.lag)
        else:
            total_num_fragments = len(
                self.trainer.train_dataloader.dataset) * (timesteps - self.model.lag)

        
        if G != []:
            G = G[0]
        F = spatial_factors
        # W = W[0]

        X_lag, Z_recon, X_recon, F_recon, G_recon, total_loss, loss_terms = self.model.compute_loss(
            X,
            total_num_fragments=total_num_fragments,
            spatial_factors=None)
        self.idx += 1 # for debugging/visualization

        return total_loss, loss_terms, G, G_recon, F, F_recon, Z_gt, Z_recon

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx: The index of the current batch.
        Returns:
            A tensor of losses between model predictions and targets.
        """

        self.validating = False # for debugging/visualization

        # model step
        total_loss, loss_terms, G, G_recon, F, F_recon, Z_gt, Z_recon = self.model_step(
            batch)
        # update and log metrics
        self.train_loss(total_loss)
        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        loss = total_loss
        
        # check for cold start
        if self.current_epoch <= self.disable_auglag_epoch:
            # loss = loss_terms['likelihood'] + loss_terms['f_term']
            for p in self.model.scm_model.parameters():
                p.requires_grad = False
            for p in self.model.temporal_graph_dist.parameters():
                p.requires_grad = False
        else:
            for p in self.model.scm_model.parameters():
                p.requires_grad = True
            for p in self.model.temporal_graph_dist.parameters():
                p.requires_grad = True

        # return loss or backpropagation will fail
        loss = self.loss_calc(loss, loss_terms['dagness_penalty'])
        loss_terms['loss'] = loss

        self.visualize = True
        return loss_terms

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx: The index of the current batch.
        Returns:
            A tensor of losses between model predictions and targets.
        """

        # model step
        loss, loss_terms, G, G_recon, F, F_recon, Z_gt, Z_recon = self.model_step(
            batch)
        P = None

        # check if ground truth graph is available
        if ((Z_gt == [] and G == []) == False) and self.model.num_nodes >= G.shape[-1]:
            if self.model.num_nodes > G.shape[-1]:
                remain = match_latent_indices(F_recon.detach().cpu().numpy(), F[0].detach().cpu().numpy())
                G_recon = G_recon[:, remain][:, :, remain]
                Z_recon = Z_recon[:,:,remain]
                F_recon = F_recon[:,remain]
            mcc, P = compute_mcc(torch.cat((Z_recon[:,0], Z_recon[-1,-self.model.lag:]), dim = 0).detach().cpu().numpy().T, 
                                    Z_gt[0].detach().cpu().numpy().T, "Pearson")
            P = P.to(self.device)
            G_recon = G_recon[:, P][:, :, P]
            print("G", G)
        
        # get predicted graph probabilities
        G_probs = self.model.temporal_graph_dist.get_adj_matrix()[
                :, P][:, :, P]
        print("G_recon", G_probs)

        # get predicted spatial factors
        if P != None:
            F_mask = F_recon[:,P,:].reshape(self.model.num_variates, -1, self.model.ny, self.model.nx)
        else:
            F_mask = F_recon[:,:,:].reshape(self.model.num_variates, self.model.num_nodes, self.model.ny, self.model.nx)

        # check if ground truth spatial factor is available
        if F != []:
            F_gt = F.reshape(self.model.num_variates, -1, self.model.ny, self.model.nx)
            # if self.model.num_nodes == G.shape[-1]:
            self.val_f_mse(F_mask,F_gt)
        
        ####################################################################################
        # TODO: Visualizations
        # save pictures
        self.validating = True
        import matplotlib.pyplot as plt
        if self.validating and batch_idx%20 == 0:
            # F_mask = F_pred[:,P,:].reshape(-1, self.model.ny, self.model.nx)
            
            for i in range(F_mask.shape[0]):
                for j in range(F_mask.shape[1]):
                    plt.imsave(os.path.join(self.save_dir, f'F_pred_{i}_{j}.png'), F_mask[i][j].detach().cpu().numpy())
            plt.close()

            # F_gt = F.reshape(-1, self.model.ny, self.model.nx)
            if F != []:
                for i in range(F_gt.shape[0]):
                    for j in range(F_gt.shape[1]):
                        plt.imsave(os.path.join(self.save_dir, f'F_gt_{i}_{j}.png'), F_gt[i][j].detach().cpu().numpy())
            plt.close()
            combined_factor = torch.sum(F_mask.detach().cpu(), axis=1).reshape(self.model.num_variates, self.model.ny, self.model.nx)
            self.F_evolve.append(combined_factor)
            self.rho_logvar.append(self.model.spatial_factors.rho_logvar.detach().cpu())
        ####################################################################################


        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        if batch_idx %100 == 1:
            for term_name, term_value in loss_terms.items():
                self.log(f"val/{term_name}", term_value)
        
        # update and log metrics if ground truth is available
        if G != [] and self.model.num_nodes >= G.shape[-1]:
            self.val_f1(G_recon, G)
            self.val_f1_lag(G_recon[1:], G[1:])
            self.val_f1_inst(G_recon[0], G[0])
            self.val_roc(G_probs, G)
    
            self.log("val/f1", self.val_f1, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("val/f1_lag", self.val_f1_lag, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("val/f1_inst", self.val_f1_inst, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("val/auroc", self.val_roc, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("val/mcc", mcc,
                    on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/f_mse", self.val_f_mse,
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

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx: The index of the current batch.
        Returns:
            A tensor of losses between model predictions and targets.
        """

        # model step
        self.test = True
        P = None
        loss, loss_terms, G, G_recon, F, F_recon, Z_gt, Z_recon = self.model_step(
            batch)

        # check if ground truth graph is available
        if ((Z_gt == [] and G == []) == False) and self.model.num_nodes >= G.shape[-1]:
            if self.model.num_nodes > G.shape[-1]:
                remain = match_latent_indices(F_recon.detach().cpu().numpy(), F[0].detach().cpu().numpy())
                G_recon = G_recon[:, remain][:, :, remain]
                Z_recon = Z_recon[:,:,remain]
            
            mcc, P = compute_mcc(torch.cat((Z_recon[:,0], Z_recon[-1,-self.model.lag:]), dim = 0).detach().cpu().numpy().T, 
                                    Z_gt[0].detach().cpu().numpy().T, "Pearson")
            P = P.to(self.device)
            G_recon = G_recon[:, P][:, :, P]
            G_probs = self.model.temporal_graph_dist.get_adj_matrix()[
                :, P][:, :, P]
        
        # get predicted spatial factors
        if P != None and self.model.num_nodes == G.shape[-1]:
            F_mask = F_recon[:,P,:].reshape(self.model.num_variates, self.model.num_nodes, self.model.ny, self.model.nx)
        else:
            F_mask = F_recon[:,:,:].reshape(self.model.num_variates, self.model.num_nodes, self.model.ny, self.model.nx)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # log metrics if ground truth is available
        if G != [] and self.model.num_nodes >= G.shape[-1]:
            self.test_f1(G_recon, G)
            self.test_f1_lag(G_recon[1:], G[1:])
            self.test_f1_inst(G_recon[0], G[0])
            self.test_roc(G_probs, G)

            
            self.log("test/f1", self.test_f1, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/f1_lag", self.test_f1_lag, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/f1_inst", self.test_f1_inst, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/auroc", self.test_roc, on_step=False,
                    on_epoch=True, prog_bar=True)
            self.log("test/mcc", mcc,
                    on_step=False, on_epoch=True, prog_bar=True)

        # Visualizations
        if batch_idx%100 == 0:
            sp = self.model.spatial_factors

            center, scale = sp.get_centers_and_scale()

            torch.save(center.detach().cpu(), os.path.join(self.save_dir, f'center.pt'))
            torch.save(scale.detach().cpu(), os.path.join(self.save_dir, f'scale.pt'))
            F_evolve = torch.stack(self.F_evolve, axis = 0)
            torch.save(F_evolve.detach().cpu(), os.path.join(self.save_dir, f'F_evolve.pt'))
            
            rho_logvar = torch.stack(self.rho_logvar, axis = 0)
            torch.save(rho_logvar.detach().cpu(), os.path.join(self.save_dir, f'rho_logvar.pt'))

            torch.save(F_mask.detach().cpu(), os.path.join(self.save_dir, f'F_pred.pt'))
            torch.save(G_recon.detach().cpu(), os.path.join(self.save_dir, f'G_pred.pt'))

            if self.model.num_variates > 1:
                alpha = self.model.alpha.sample_alpha()
                torch.save(alpha.detach().cpu(), os.path.join(self.save_dir, f'alpha.pt'))

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
        """Lightning hook that is called after the backward pass."""
        for p in self.parameters():
            if p.grad != None:
                p.grad = torch.nan_to_num(p.grad)
        return super().on_after_backward()

    def configure_callbacks(self):
        """Create a callback for the auglag callback."""
        return [AuglagLRCallback(self.lr_scheduler, log_auglag=True, disabled_epochs=self.disable_auglag_epoch)]
