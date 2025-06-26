from dataclasses import dataclass
from typing import List
from itertools import product


@dataclass
class SyntheticDataConfig:
    """Config class to describe the properties of a generated synthetic dataset

    """

    # data properties
    model: str
    time_length: int

    # grid properties
    nx: int
    ny: int
    node_dist: int
    node_extent_low: int
    node_extent_high: int
    random_node: bool
    num_variates: int
    disjoint_nodes: bool
    map_type: str
    grid_noise: str
    grid_noise_scale: float

    # graph properties
    functional_relationships: str
    inst_graph_type: str
    lag_graph_type: str
    lag: int
    num_nodes: int
    base_noise_type: str
    hist_dep_noise_type: str
    noise_scale: float
    hist_dep_noise: bool
    disable_inst: bool

    # optional
    seed: int
