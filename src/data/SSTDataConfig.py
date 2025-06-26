from dataclasses import dataclass
from typing import List
from itertools import product


@dataclass
class SSTDataConfig:
    """Config class to describe the properties of a real SST dataset

    """

    # data properties
    model: str
    time_length: int

    # grid properties
    nx: int
    ny: int
    num_variates: int
    map_type: str
    
    # graph properties
    functional_relationships: str
    lag: int
    num_nodes: int

    # optional
    seed: int

        # Deprecated
    # grid_noise: str
    # grid_noise_scale: float
    # node_dist: int
    # node_extent_low: int
    # node_extent_high: int
    # random_node: bool
    # disjoint_nodes: bool
    # inst_graph_type: str
    # lag_graph_type: str
    # base_noise_type: str
    # hist_dep_noise_type: str
    # noise_scale: float
    # hist_dep_noise: bool
    # disable_inst: bool