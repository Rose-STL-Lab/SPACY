"""
Acknowledgements: Lot of the code borrowed from https://github.com/microsoft/causica
and from https://github.com/xtibau/mapped_pcmci
"""

import numpy as np
import torch
import random
import igraph as ig

from typing import Callable, Dict, List, Optional, Tuple
from src.utils.data_generation.syn_functions import sample_spline, sample_spline_product, \
    sample_conditional_spline, sample_inverse_noise_spline, sample_linear, \
    sample_mlp, sample_mlp_noise, sample_none, zero_func, sample_gaussian_linear


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_function(input_dim: int, function_type: str) -> Callable:
    """
    This will sample a function given function type.
    Args:
        input_dim: The input dimension of the function.
        function_type: The type of function to be sampled.
    Returns:
        A function sampled.
    """
    if function_type == "spline":
        return sample_spline(input_dim)
    if function_type == "spline_product":
        return sample_spline_product(input_dim)
    elif function_type == "conditional_spline":
        return sample_conditional_spline(input_dim)
    elif function_type == "mlp":
        return sample_mlp(input_dim)
    elif function_type == "inverse_noise_spline":
        return sample_inverse_noise_spline(input_dim)
    elif function_type == "mlp_noise":
        return sample_mlp_noise(input_dim)
    elif function_type == 'linear':
        return sample_linear(input_dim)
    elif function_type == "none":
        return sample_none(input_dim)
    elif function_type == "gaus_linear":
        return sample_gaussian_linear(input_dim)
    else:
        raise ValueError(f"Unsupported function type: {function_type}")


def random_permutation(M: np.ndarray) -> np.ndarray:
    """
    This will randomly permute the matrix M.
    Args:
        M: the input matrix with shape [num_node, num_node].

    Returns:
        The permuted matrix
    """
    P = np.random.permutation(np.eye(M.shape[0]))
    return P.T @ M @ P


def random_acyclic_orientation(B_und: np.ndarray) -> np.ndarray:
    """
    This will randomly permute the matrix B_und followed by taking the lower triangular part.
    Args:
        B_und: The input matrix with shape [num_node, num_node].

    Returns:
        The lower triangular of the permuted matrix.
    """
    return np.tril(random_permutation(B_und), k=-1)


def generate_single_graph(num_nodes: int, graph_type: str, graph_config: dict, is_DAG: bool = True) -> np.ndarray:
    """
    This will generate a single adjacency matrix following different graph generation methods (specified by graph_type, can be "ER", "SF", "SBM").
    graph_config specifes the additional configurations for graph_type. For example, for "ER", the config dict keys can be {"p", "m", "directed", "loop"},
    refer to igraph for details. is_DAG is to ensure the generated graph is a DAG by lower-trianguler the adj, followed by a permutation.
    Note that SBM will no longer be a proper SBM if is_DAG=True
    Args:
        num_nodes: The number of nodes
        graph_type: The graph generation type. "ER", "SF" or "SBM".
        graph_config: The dict containing additional argument for graph generation.
        is_DAG: bool indicates whether the generated graph is a DAG or not.

    Returns:
        An binary ndarray with shape [num_node, num_node]
    """
    if graph_type == "ER":
        adj_graph = np.array(ig.Graph.Erdos_Renyi(
            n=num_nodes, **graph_config).get_adjacency().data)
        if is_DAG:
            adj_graph = random_acyclic_orientation(adj_graph)
    elif graph_type == "SF":
        if is_DAG:
            graph_config["directed"] = True
        adj_graph = np.array(ig.Graph.Barabasi(
            n=num_nodes, **graph_config).get_adjacency().data)
    elif graph_type == "SBM":
        adj_graph = np.array(ig.Graph.SBM(
            n=num_nodes, **graph_config).get_adjacency().data)
        if is_DAG:
            adj_graph = random_acyclic_orientation(adj_graph)
    else:
        raise ValueError("Unknown graph type")

    if is_DAG or graph_type == "SF":
        # SF needs to be permuted otherwise it generates either lowtri or symmetric matrix.
        adj_graph = random_permutation(adj_graph)

    return adj_graph


def generate_temporal_graph(num_nodes: int,
                            graph_type: List[str],
                            graph_config: List[dict],
                            lag: int,
                            random_state: Optional[int] = None
                            ) -> np.ndarray:
    """
    This will generate a temporal graph with shape [lag+1, num_nodes, num_nodes] based on the graph_type. The graph_type[0] specifies the
    generation type for instantaneous effect and graph_type[1] specifies the lagged effect. For re-produciable results, set random_state.
    Args:
        num_nodes: The number of nodes.
        graph_type: A list containing the graph generation type. graph_type[0] for instantaneous effect and graph_type[1] for lagged effect.
        graph_config: A list of dict containing the configs for graph generation. The order should respect the graph_type.
        lag: The lag of the graph.
        random_state: The random seed used to generate the graph. None means not setting the seed.

    Returns:
        temporal_graph with shape [lag+1, num_nodes, num_nodes]
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    temporal_graph = np.full([lag + 1, num_nodes, num_nodes], np.nan)
    # Generate instantaneous effect graph
    temporal_graph[0] = generate_single_graph(
        num_nodes, graph_type[0], graph_config[0], is_DAG=True)
    # Generate lagged effect graph
    for i in range(1, lag + 1):
        temporal_graph[i] = generate_single_graph(
            num_nodes, graph_type[1], graph_config[1], is_DAG=False)

    return temporal_graph


def build_function_list(
    temporal_graph: np.ndarray, function_type: str, noise_function_type: str
) -> Tuple[List[Callable], List[Callable]]:
    """
    This will build two lists containing the SEM functions and history-dependent noise function, respectively.
    Args:
        temporal_graph: The input temporal graph.
        function_type: The type of SEM function used.
        noise_function_type: The tpe of history-dependent noise transformation used.

    Returns:
        function_list: list of SEM functions
        noise_function_list: list of history-dependent noise transformation
    """
    num_nodes = temporal_graph.shape[1]
    # get func_list
    function_list = []
    for cur_node in range(num_nodes):
        # get input dim
        input_dim = sum(temporal_graph[lag, :, cur_node] for lag in range(
            temporal_graph.shape[0])).sum().astype(int)  # type: ignore

        if input_dim == 0:
            function_list.append(zero_func)
        else:
            function_list.append(sample_function(
                input_dim, function_type=function_type))
    # get noise_function_list
    noise_function_list = []
    for cur_node in range(num_nodes):
        # get input dim
        input_dim = (
            sum(temporal_graph[lag, :, cur_node] for lag in range(
                1, temporal_graph.shape[0])).sum() + 1  # type: ignore
        ).astype(int)
        noise_function_list.append(sample_function(
            input_dim, function_type=noise_function_type))

    return function_list, noise_function_list


def extract_parents(data: np.ndarray, temporal_graph: np.ndarray, node: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will extract the parent values from data with given graph temporal_graph. It will return the lagged parents
    and instantaneous parents.
    Args:
        data: ndarray with shape [series_length, num_nodes] or [batch, series_length, num_nodes]
        temporal_graph: A binary ndarray with shape [lag+1, num_nodes, num_nodes]

    Returns:
        instant_parent: instantaneous parents with shape [parent_dim] or [batch, parents_dim]
        lagged_parent: lagged parents with shape [lagged_dim] or [batch, lagged_dim]
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # shape [1, series_length, num_nodes]

    assert data.ndim == 3, "data should be of shape [series_length, num_nodes] or [batch, series_length, num_nodes]"
    lag = temporal_graph.shape[0] - 1
    # extract instantaneous parents
    inst_parent_node = temporal_graph[0, :, node].astype(
        bool)  # shape [num_parents]
    # shape [batch, parent_dim]
    inst_parent_value = data[:, -1, inst_parent_node]

    # Extract lagged parents
    lagged_parent_value_list = []
    for cur_lag in range(1, lag + 1):
        cur_lag_parent_node = temporal_graph[cur_lag, :, node].astype(
            bool)  # shape [num_parents]
        # shape [batch, parent_dim]
        cur_lag_parent_value = data[:, -cur_lag - 1, cur_lag_parent_node]
        lagged_parent_value_list.append(cur_lag_parent_value)

    # shape [batch, lagged_dim_aggregated]
    lagged_parent_value = np.concatenate(lagged_parent_value_list, axis=1)

    # if data.shape[0] == 1:
    #     inst_parent_value, lagged_parent_value = inst_parent_value.squeeze(
    #         0), lagged_parent_value.squeeze(0)

    return inst_parent_value, lagged_parent_value


def simulate_history_dep_noise(lagged_parent_value: np.ndarray, noise: np.ndarray, noise_func: Callable) -> np.ndarray:
    """
    This will simulate the history-dependent noise given the lagged parent value.
    Args:
        lagged_parent_value: ndarray with shape [batch, lag_parent_dim] or [lag_parent_dim]
        noise: ndarray with shape [batch,1] or [1]
        noise_func: this specifies the function transformation for noise

    Returns:
        history-dependent noise with shape [batch, 1] or [1]
    """

    assert (
        lagged_parent_value.shape[0] == noise.shape[0]
    ), "lagged_parent_value and noise should have the same batch size"
    if lagged_parent_value.ndim == 1:
        # shape [1, lag_parent_dim]
        lagged_parent_value = lagged_parent_value[np.newaxis, ...]
        noise = noise[np.newaxis, ...]  # [1, 1]

    # concat the lagged parent value and noise
    # shape [batch, lag_parent_dim+1]
    input_to_gp = np.concatenate([lagged_parent_value, noise], axis=1)
    history_dependent_noise = noise_func(input_to_gp)  # shape [batch, 1]

    if lagged_parent_value.shape[0] == 1:
        history_dependent_noise = history_dependent_noise.squeeze(0)

    return history_dependent_noise


def simulate_function(lag_inst_parent_value: np.ndarray, func: Callable) -> np.ndarray:
    """
    This will simulate the function given the lagged and instantaneous parent values. The random_state_value controls
    which function is sampled from gp_func, and it controls which function is used for a particular node.
    Args:
        lag_inst_parent_value: ndarray with shape [batch, lag+inst_parent_dim] or [lag+inst_parent_dim]
        random_state_value: int controlling the function sampled from gp
        func: This specifies the functional relationships

    Returns:
        ndarray with shape [batch, 1] or [1] representing the function value for the current node.
    """

    if lag_inst_parent_value.ndim == 1:
        # shape [1, lag+inst_parent_dim]
        lag_inst_parent_value = lag_inst_parent_value[np.newaxis, ...]

    function_value = func(lag_inst_parent_value)  # shape [batch, 1]

    if lag_inst_parent_value.shape[0] == 1:
        function_value = function_value.squeeze(0)

    return function_value


def simulate_single_step(
    history_data: np.ndarray,
    temporal_graph: np.ndarray,
    func_list_noise: List[Callable],
    func_list: List[Callable],
    topological_order: List[int],
    is_history_dep: bool = False,
    noise_level: float = 1,
    base_noise_type: str = "gaussian",
    intervention_dict: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    This will generate the data for a particular timestamp given the history data and temporal graph.
    Args:
        history_data: History data with shape [batch, series_length, num_node] or [series_length, num_node] containing the history observations.
        temporal_graph: The binary ndarray graph with shape [lag+1, num_nodes, num_nodes].
        func_list_noise: List of function transforms for the noise variable. Shape [num_nodes]. Each function takes [batch, dim] as input and
        output [batch,1]
        func_list: List of function for each variable, shape [num_nodes]. Each func takes [batch, dim] as input and output [batch,1]
        topological_order: the topological order from source to leaf nodes, specified by temporal graph.
        is_history_dep: bool indicate if the noise is history dependent
        base_noise_type: str, support "gaussian" and "uniform"
        intervention_dict: dict holding interventions for the current time step of form {intervention_idx: intervention_value}
    Returns:
        ndarray with shape [batch,num_node] or [num_node]
    """

    assert (
        len(func_list_noise) == len(func_list) == temporal_graph.shape[1]
    ), "function list and topological_order should have the same length as the number of nodes"
    if history_data.ndim == 2:
        # shape [1, series_length, num_nodes]
        history_data = history_data[np.newaxis, ...]

    if intervention_dict is None:
        intervention_dict = {}

    batch_size = history_data.shape[0]
    # iterate through the nodes in topological order
    for node in topological_order:
        # if this node is intervened on - ignore the functions and parents
        if node in intervention_dict:
            history_data[:, -1, node] = intervention_dict[node]
        else:
            # extract the instantaneous and lagged parent values
            inst_parent_value, lagged_parent_value = extract_parents(
                history_data, temporal_graph, node
            )  # [batch, inst_parent_dim], [batch, lagged_dim_aggregated]

            # simulate the noise
            if base_noise_type == "gaussian":
                Z = np.random.randn(history_data.shape[0], 1)  # [batch, 1]
            elif base_noise_type == "uniform":
                Z = np.random.rand(history_data.shape[0], 1)
            elif base_noise_type == "none":
                Z = np.zeros((history_data.shape[0], 1))
            elif base_noise_type == 'exponential':
                Z = np.random.exponential(scale=1.0, size=(history_data.shape[0], 1)) - 1
            else:
                raise NotImplementedError
            if is_history_dep:
                # Extract the noise transform
                noise_func = func_list_noise[node]

                Z = simulate_history_dep_noise(
                    lagged_parent_value, Z, noise_func)  # [batch, 1]

            # simulate the function
            lag_inst_parent_value = np.concatenate(
                [inst_parent_value, lagged_parent_value], axis=1
            )  # [batch, lag+inst_parent_dim]
            if lag_inst_parent_value.size == 0:
                func_value = np.array(0.0)
            else:
                # Extract the function relation
                func = func_list[node]
                func_value = simulate_function(lag_inst_parent_value, func)
            # [batch]
            history_data[:, -1,
                         node] = (func_value + noise_level * Z).squeeze(-1)

    X = history_data[:, -1, :]  # [batch,num_nodes]
    if batch_size == 1:
        X = X.squeeze(0)
    return X
