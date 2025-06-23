from typing import List, Callable
import numpy as np
import igraph as ig
from functools import partial
from src.utils.data_generation.graph_utils import simulate_single_step
import tqdm
import torch
import math

class SAVAR:
    """
    Main class containing SAVAR model
    """

    def __init__(self,
                 num_samples: int,
                 lag: int,
                 time_length: int,
                 mode_weights: np.ndarray,
                 graph: np.ndarray,
                 func_list: List[Callable],
                 noise_func_list: List[Callable],
                 burnin_steps: int = 100,
                 noise_strength: float = 1,
                 latent_noise_cov: np.ndarray = None,
                 fast_cov: np.ndarray = None,
                 history_dep_noise: bool = False
                 ):

        self.graph = graph

        self.num_samples = num_samples
        self.time_length = time_length
        self.burnin_steps = burnin_steps
        self.noise_strength = noise_strength

        # TODO: Expand/remove
        self.latent_noise_cov = latent_noise_cov  # D_x
        self.fast_noise_cov = fast_cov  # D_y

        self.mode_weights = mode_weights
        self.history_dep_noise = history_dep_noise

        # Computed attributes
        self.num_variates = mode_weights.shape[0]
        self.num_nodes = mode_weights.shape[1]
        self.nx = mode_weights.shape[2]
        self.ny = mode_weights.shape[3]

        self.num_grid_points = self.nx * self.ny

        self.lag = lag

        self.noise_weights = self.mode_weights

        if self.latent_noise_cov is None:
            self.latent_noise_cov = np.eye(self.num_nodes)
        # if self.fast_noise_cov is None:
        #     # TODO: Should be np.eye ???
        #     self.fast_noise_cov = np.zeros(
        #         (self.num_variates, self.num_grid_points, self.num_grid_points))

        self.func_list = func_list
        self.noise_func_list = noise_func_list

    def generate_data(self) -> None:
        """
        Generates the data of savar
        :return:
        """

        data = np.zeros(
            (self.num_samples, self.num_variates, self.time_length + self.burnin_steps, self.num_grid_points))

        # Add noise first
        data += self._generate_noise()

        return self.run_savar(data)

    def get_wplus(self) -> np.ndarray:
        """
        W \in NxL
        data_field L times T

        :return:
        """
        W = self.noise_weights.reshape(
            self.num_variates, self.num_nodes, self.num_grid_points)
        W_plus = np.linalg.pinv(W)

        return W_plus

        # cov = self.noise_strength * \
        # W_plus @ W_plus.transpose((0, 2, 1))

        # we may or may not want to add the following line
        # + 0.1 * np.repeat(np.eye(self.num_grid_points)[np.newaxis, :], self.num_variates, axis=0)

    def _generate_noise(self):

        w_plus = self.get_wplus()
        print("Generating noise...")
        # Generate noise from cov
        
        noise = math.sqrt(self.noise_strength) * np.random.randn(self.num_samples, self.num_variates,
                                self.time_length+self.burnin_steps, self.num_nodes)
        
        noise = np.einsum("kln,bktn->bktl", w_plus, noise)
        # # TODO: come back
        # w_plus = torch.tensor(w_plus).to('cuda:0').double()
        # noise = torch.randn(self.num_samples, self.num_variates,
        #                     self.time_length+self.burnin_steps, self.num_nodes, device='cuda:0').double()

        # noise = torch.einsum("kln,bktn->bktl", w_plus, noise).detach().cpu().numpy()
        #######################################################################
        # TODO: remove once we make sure this is not needed
        # noise = np.random.randn(self.num_samples, self.num_variates,
        #                         self.time_length+self.burnin_steps, self.num_grid_points)
        # noise_cov  = self.noise_strength * w_plus @ w_plus.transpose((0, 2, 1))

        # for i in range(self.num_variates):
        #     noise[:, i] = np.random.multivariate_normal(mean=np.zeros((self.num_grid_points)),
        #                                                 cov=noise_cov[i],
        #                                                 size=(self.num_samples, self.time_length+self.burnin_steps))

        print("Done!")
        return noise

    def run_savar(self, X):

        weights = self.mode_weights.reshape(
            self.num_variates, self.num_nodes, -1)
        weights_inv = np.linalg.pinv(weights)

        func_list = self.func_list
        noise_func_list = self.noise_func_list
        # Find topological order of instant graph
        ig_graph = ig.Graph.Adjacency(self.graph[0].tolist())
        topological_order = ig_graph.topological_sorting()
        single_step = partial(
            simulate_single_step,
            temporal_graph=self.graph,
            func_list=func_list,
            func_list_noise=noise_func_list,
            topological_order=topological_order,
            is_history_dep=self.history_dep_noise,
            noise_level=self.noise_strength,
            base_noise_type='gaussian'
        )

        # Start data gen
        for time in tqdm.trange(self.lag+1, self.burnin_steps + self.time_length):
            # first project data to latent space
            X_history = np.einsum("knl,bktl->btn", weights,
                                  X[:, :, time-self.lag:time+1])

            # pass through the SCM
            X_new = single_step(history_data=X_history)
            # project back to grid space
            X[:, :, time] += np.einsum("kln,bn->bkl", weights_inv, X_new)
        return X[:, :, self.burnin_steps:]
