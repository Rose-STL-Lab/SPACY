import os
import torch
import tables
import numpy as np
from typing import Tuple
from geopy import distance


def sample_data(batch, params, valid: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_size: the number of examples in a minibatch
            valid: if True, sample from validation set
        Returns:
            x, y, z: tensors of the data x, the data to predict y, and the latent variables z
        """
        X, G, Z_gt, spatial_factors = batch
        tau = params['tau']
        d = params['num_nodes']
        d_x = params['grid_size']
    
        x = np.zeros((batch_size, tau, d, d_x))
        if Z_gt != None:
            z = np.zeros((batch_size, tau + 1, d, d_z))
        else:
            z = None
        t1 = 0
        y = np.zeros((batch_size, self.d, self.d_x))

        # if valid:
        #     dataset_idx = self.idx_valid
        # else:
        #     dataset_idx = self.idx_train

        if self.n == 1:
            # if there is only one long timeserie
            random_idx = np.random.choice(dataset_idx, replace=False, size=batch_size)
            for i, idx in enumerate(random_idx):
                x[i] = self.x[0, idx - self.tau:idx + t1]
                y[i] = self.x[0, idx + t1]
                if not self.no_gt and self.latent:
                    z[i] = self.z[0, idx - self.tau:idx + t1 + 1]
        else:
            # if there are multiple timeseries
            random_idx = np.random.choice(dataset_idx, replace=False, size=batch_size)
            for i, idx in enumerate(random_idx):
                x[i] = self.x[idx, 0:self.tau + t1]
                y[i] = self.x[idx, self.tau + t1]
                if not self.no_gt and self.latent:
                    z[i] = self.z[idx, 0:self.tau + t1 + 1]

        # convert to torch tensors
        x_ = torch.tensor(x)
        y_ = torch.tensor(y)
        if not self.no_gt and self.latent:
            z_ = torch.tensor(z)
        else:
            z_ = z

        return x_, y_, z_