"""
Acknowledgements: Lot of the code borrowed from https://github.com/microsoft/causica
and from https://github.com/xtibau/mapped_pcmci
"""

import dis
from typing import List
import numpy as np
import torch
import itertools
import tqdm
import random
from src.utils.files import generate_datafolder_name, make_dirs
from src.data.SyntheticDataConfig import SyntheticDataConfig
from src.utils.data_generation.graph_utils import generate_temporal_graph, build_function_list
from src.utils.data_generation.grid_utils import generate_grid, generate_random_placement, generate_savar
import os
import matplotlib.pyplot as plt
import seaborn as sns


class SyntheticDataFactory:
    """
    Creates synthetic data
    :param num_samples: An int specifying how many samples to generate per setting.
    :param data_config: An object of type SyntheticDataConfig which specifies the kind of data to generate
    :param save_dir: A str containing the path to the save directory for the data.
    :param burnin_steps: An int denoting the burn in period for each sample.
    :param gpu: An int specifying which gpu to use. If set to -1, no gpu is used
    """

    def __init__(self,
                 num_samples: int,
                 data_config: SyntheticDataConfig,
                 save_dir: str,
                 burnin_steps: int,
                 gpu: int = -1
                 ):

        self.num_samples = num_samples
        self.data_config = data_config
        self.model = data_config.model
        self.save_dir = save_dir
        self.burnin_steps = burnin_steps
        self.device = f"cuda:{gpu}" if gpu != -1 else "cpu"

        assert self.burnin_steps >= 0, "Burn-in period should be >= 0"

    def run(self):

        cfg = self.data_config

        # generate graph
        connection_factor = 1
        graph_type = [cfg.inst_graph_type, cfg.lag_graph_type]
        graph_config = [
            {"m": int(random.randint(1, cfg.num_nodes) * 2 * connection_factor)
             if not cfg.disable_inst else 0, "directed": True},
            {"m": int(cfg.num_nodes * connection_factor),
             "directed": True},
        ]
        graph = generate_temporal_graph(num_nodes=cfg.num_nodes,
                                        graph_type=graph_type,
                                        graph_config=graph_config,
                                        lag=cfg.lag)
        
        # generate node placement
        node_placement = generate_random_placement(
            num_nodes=cfg.num_nodes,
            nx=cfg.nx,
            ny=cfg.ny,
            node_dist=cfg.node_dist,
            num_variates=cfg.num_variates,
            disjoint=cfg.disjoint_nodes)

        stable = False

        while stable == False:
            # build the function list
            func_list, noise_func_list = build_function_list(
                graph, function_type=cfg.functional_relationships, noise_function_type=cfg.hist_dep_noise_type
            )
            ########################################################################
            if cfg.model == "savar":
                X, mode_weights, mapping = generate_savar(cfg=cfg,
                                                    num_samples=self.num_samples,
                                                    graph=graph,
                                                    node_placement=node_placement,
                                                    func_list=func_list,
                                                    noise_func_list=noise_func_list,
                                                    burnin_steps=self.burnin_steps)
                return X, graph, mode_weights, mapping, node_placement
            ########################################################################

            X, z_latent, spatial_factors = generate_grid(cfg=cfg,
                                                    num_samples=self.num_samples,
                                                    graph=graph,
                                                    node_placement=node_placement,
                                                    func_list=func_list,
                                                    noise_func_list=noise_func_list,
                                                    burnin_steps=self.burnin_steps)
            
            X = X - np.mean(X, axis = (2,3))[:,:,np.newaxis,np.newaxis]

            grid_wise_range = np.max(X, axis = (2,3)) - np.min(X, axis =(2,3))
            if np.max(grid_wise_range) < 1e2 and np.min(grid_wise_range) > 1.0:
                if np.max(z_latent[:,:,0])-np.min(z_latent[:,:,0]) > 2:
                    stable = True

        return X, graph, z_latent, spatial_factors, node_placement

    def save_data(self, X, graph, z_latent, spatial_factors, node_placement):
        """
        This function saves data with specified parameters.

        :param X: np.ndarray
            The data array.
        :param graph: np.ndarray
            The graph adjacency matrix.
        :param z_latent: np.ndarray
            The latent data
        :param spatial_factors: np.ndarray
            Spatial Factors 
        :param node_placement: np.ndarray
            Where each node is placed
        """

        cfg = self.data_config
        # Save data with specified parameters
        folder_name = generate_datafolder_name(cfg)
        folder_path = os.path.join(self.save_dir, folder_name)
        make_dirs(folder_path)

        # save each file separately
        for i in range(self.num_samples):
            Xi = torch.from_numpy(X[i])
            torch.save(Xi, os.path.join(folder_path, f"{i}.pt"))
            Zi = torch.from_numpy(z_latent[i])
            torch.save(Zi, os.path.join(folder_path, f"{i}_latent.pt"))

        torch.save(torch.from_numpy(graph),
                   os.path.join(folder_path, f"graph.pt"))
        torch.save(torch.from_numpy(spatial_factors),
                   os.path.join(folder_path, f"spatial_factors.pt"))


    def generate_data(self):
        """
        Generate synthetic data based on the specified parameters and saves it to the save directory
        """

        # TODO: implement
        # if self.data_config.hist_dep_noise:
        #     raise NotImplementedError

        if self.model == "savar":
            X, G, mode_weights, mapping, node_placement = self.run()
            self.save_savar(X=X,
                       graph=G,
                       mode_weights=mode_weights,
                       mapping=mapping,
                       node_placement=node_placement
                       )
            
        else:
            X, G, z_latent, spatial_factors, node_placement = self.run()

            self.save_data(X=X,
                        graph=G,
                        z_latent=z_latent,
                        spatial_factors=spatial_factors,
                        node_placement=node_placement
                        )
        print('Data Generated and Saved')

    def save_savar(self, X, graph, mode_weights, mapping, node_placement):
        """
        This function saves data with specified parameters.

        :param X: np.ndarray
            The data array.
        :param graph: np.ndarray
            The graph adjacency matrix.
        :param mode_weights: np.ndarray
            Mode Weights
        :param mapping: np.ndarray
            Spatial mapping 
        :param node_placement: np.ndarray
            Where each node is placed
        """

        cfg = self.data_config
        # Save data with specified parameters
        folder_name = generate_datafolder_name(cfg)
        folder_path = os.path.join(self.save_dir, folder_name)
        make_dirs(folder_path)

        # save each file separately
        for i in range(self.num_samples):
            Xi = torch.from_numpy(X[i])
            torch.save(Xi, os.path.join(folder_path, f"{i}.pt"))

        torch.save(torch.from_numpy(graph),
                   os.path.join(folder_path, f"graph.pt"))
        torch.save(torch.from_numpy(mode_weights),
                   os.path.join(folder_path, f"mode_weights.pt"))
        torch.save(torch.from_numpy(mapping),
                   os.path.join(folder_path, f"spatial_mapping.pt"))

        # save images
        for i in range(mapping.shape[0]):
            plt.figure()
            sns.heatmap(mapping[i].reshape(cfg.ny, cfg.nx), cbar=False)
            plt.savefig(os.path.join(folder_path, f'variate_{i}.png'))
            plt.close()

        for i in range(mapping.shape[0]):
            plt.figure()
            plt.imsave(os.path.join(folder_path, f'imvariate_{i}.png'),
                       mapping[i].reshape(cfg.ny, cfg.nx))
            plt.close()

        for i in range(mode_weights.shape[0]):
            plt.figure()
            sns.heatmap(np.sum(mode_weights[i], axis=0).reshape(
                cfg.ny, cfg.nx), cbar=False)
            plt.savefig(os.path.join(
                folder_path, f'variate_{i}_modeweights.png'))
            plt.close()
