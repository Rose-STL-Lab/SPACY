from typing import List, Callable
import numpy as np
import igraph as ig
from functools import partial
from src.utils.data_generation.graph_utils import simulate_single_step
import tqdm
import torch
import math

class LATENT:
    """
    Main class containing Latent data generation
    """

    def __init__(self,
                 num_samples: int,
                 num_nodes: int,
                 lag: int,
                 time_length: int,
                 graph: np.ndarray,
                 func_list: List[Callable],
                 noise_func_list: List[Callable],
                 burnin_steps: int = 100,
                 base_noise_type: str = 'gaussian',
                 noise_scale: float = 1,
                 latent_noise_cov: np.ndarray = None,
                 fast_cov: np.ndarray = None,
                 history_dep_noise: bool = False
                 ):

        self.graph = graph

        self.num_samples = num_samples
        self.time_length = time_length
        self.burnin_steps = burnin_steps
        self.base_noise_type = base_noise_type
        self.noise_scale = noise_scale

        # TODO: Expand/remove
        self.latent_noise_cov = latent_noise_cov  # D_x
        self.fast_noise_cov = fast_cov  # D_y

        self.history_dep_noise = history_dep_noise

        # Computed attributes
        self.num_nodes = num_nodes


        self.lag = lag


        if self.latent_noise_cov is None:
            self.latent_noise_cov = np.eye(self.num_nodes)
        # if self.fast_noise_cov is None:
        #     # TODO: Should be np.eye ???
        #     self.fast_noise_cov = np.zeros(
        #         (self.num_variates, self.num_grid_points, self.num_grid_points))

        self.func_list = func_list
        self.noise_func_list = noise_func_list

    def generate_data(self) -> None:
        """Generates the data of savar
        Returns:
            X: np.ndarray
                The generated data of shape (num_samples, time_length, num_nodes)
        """
        # X = np.zeros((self.num_samples, self.time_length + self.burnin_steps, self.num_nodes))

        X = np.zeros((self.num_samples, self.burnin_steps + self.time_length + self.lag, self.num_nodes))
        X[..., 0:self.lag+1, :] = (
            np.random.randn(self.num_samples, self.lag+1,
                            self.num_nodes) if self.num_samples > 1 else np.random.randn(self.lag+1, self.num_nodes)
        )

        return self.run_generation(X)


    def run_generation(self, X):
        """ This function runs the generation of the data
        Args:
            X: np.ndarray
                The generated data of shape (num_samples, time_length + burnin_steps, num_nodes)
        Returns:
            X: np.ndarray
                The generated data of shape (num_samples, time_length, num_nodes)
        """

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
            noise_level=self.noise_scale,
            base_noise_type=self.base_noise_type
        )
    
        # Start data gen
        for time in tqdm.trange(self.lag, self.burnin_steps + self.time_length + self.lag):
            # first project data to latent space
            X_history = X[:, time-self.lag:time+1]

            # pass through the SCM
            X_new = single_step(history_data=X_history)
            # project back to grid space
            X[:, time] = X_new
        return X[:, self.burnin_steps+self.lag:]

