import numpy as np
from scipy import spatial
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from src.utils.data_generation.LATENT import LATENT
from src.model.modules.decoding.SpatialDecoderNN import SpatialDecoderNN

## Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ## General Methods


def generate_random_placement(num_nodes, nx, ny, node_dist, num_variates, disjoint=False):
    """Generates random placement of nodes in the grid, favoring center locations.

    Args:
        num_nodes (int): Number of nodes in the system
        nx (int): Number of rows
        ny (int): Number of columns
        node_dist (int): Min distance between center of each node 
        num_variates (int): Number of variates (distinct grids) to generate
        case (int): 
            1 = disjoint nodes across variates, 
            2 = nodes have presence across variates. 
            Defaults to 1.

    Returns:
        node_placement: List of locations in each variate where nodes are placed
    """
    
    node_placement = [[None for _ in range(num_nodes)] for _ in range(num_variates)]

    # Generate node placement for both cases
    if disjoint:  # Disjoint nodes across variates
        nodes_per_variate = num_nodes // num_variates
        node_var_assignment = np.repeat(np.arange(num_variates), nodes_per_variate)
        np.random.shuffle(node_var_assignment)
        
        # Generate random locations for each node and assign them to variates
        min_distance = node_dist
        center_x = nx / 2
        center_y = ny / 2
        std_dev_x = nx / 4  # Controls the spread along the x-axis
        std_dev_y = ny / 4  # Controls the spread along the y-axis

        for i in range(num_nodes):
            placed = False
            while not placed:
                # Propose a random placement for the new node using a normal distribution centered in the grid
                new_node_x = np.random.normal(center_x, std_dev_x)
                new_node_y = np.random.normal(center_y, std_dev_y)

                # Ensure the node is within the grid boundaries
                new_node_x = np.clip(new_node_x, 0, nx - node_dist)
                new_node_y = np.clip(new_node_y, 0, ny - node_dist)

                new_node = [new_node_x, new_node_y]

                # Check if the new node is at least min_distance away from all other nodes in the same variate
                overlap = False
                for existing_node in node_placement[node_var_assignment[i]]:
                    if existing_node is None:
                        continue
                    distance = np.sqrt((existing_node[0] - new_node[0]) ** 2 +
                                       (existing_node[1] - new_node[1]) ** 2)
                    if distance < min_distance:
                        overlap = True
                        break

                # If no overlap, place the new node
                if not overlap:
                    node_placement[node_var_assignment[i]][i] = new_node
                    placed = True

    elif disjoint == False:  # Nodes present across variates
        # Generate one set of random node locations and assign to all variates
        min_distance = node_dist
        center_x = nx / 2
        center_y = ny / 2
        std_dev_x = nx / 4  # Controls the spread along the x-axis
        std_dev_y = ny / 4  # Controls the spread along the y-axis

        for i in range(num_variates):
            for j in range(num_nodes):
                placed = False
                while not placed:
                    # Propose a random placement for the new node using a normal distribution centered in the grid
                    new_node_x = np.random.normal(center_x, std_dev_x)
                    new_node_y = np.random.normal(center_y, std_dev_y)

                    # Ensure the node is within the grid boundaries
                    new_node_x = np.clip(new_node_x, 0, nx - node_dist)
                    new_node_y = np.clip(new_node_y, 0, ny - node_dist)

                    new_node = [new_node_x, new_node_y]

                    # Check if the new node is at least min_distance away from all other nodes
                    overlap = False
                    for existing_node in node_placement[i]:  # check for overlap on the i-th variate
                        if existing_node is None:
                            continue
                        distance = np.sqrt((existing_node[0] - new_node[0]) ** 2 +
                                        (existing_node[1] - new_node[1]) ** 2)
                        if distance < min_distance:
                            overlap = True
                            break

                    # If no overlap, assign the node to the variate
                    if not overlap:
                        node_placement[i][j] = new_node
                        placed = True

    return node_placement

## Complex kernel


def irregular_kernel(grid_coords, center, extent, warp_amp, warp_freq, simple=True):
    """
    This function computes an irregular kernel values for a given set of grid coordinates    
    First, the coordinates (x, y) are distorted via a sinusoidal warp.
    Then, an RBF kernel is evaluated on the warped coordinates using the specified extent.
    
    Args:
        grid_coords (array-like): Coordinates array with shape (..., 2).
        center (array-like): Kernel center coordinates, shape (2,).
        extent: Parameter controlling the Gaussian's extent. If simple=True, a scalar or array-like of shape (2,); else, a tuple (A, B).
        warp_amp (float): Amplitude of the sinusoidal warp.
        warp_freq (float): Frequency of the sinusoidal warp.
        simple (bool, optional): If True, use a diagonal covariance matrix. If False, use a full covariance matrix.
        
    Returns:
        array: Kernel values as a 2D array.
    """
    # Extract x and y components from grid_coords
    x = grid_coords[..., 0]
    y = grid_coords[..., 1]
    x0, y0 = center
    
    # Apply sinusoidal warp to the coordinates
    dx = x - x0
    dy = y - y0
    u = x + warp_amp * np.sin(warp_freq * dx) * np.sin(warp_freq * dy)
    v = y + warp_amp * np.cos(warp_freq * dx) * np.cos(warp_freq * dy)
    
    # Compute warped differences from the center
    diff_warped = np.stack([u - x0, v - y0], axis=-1)
    # Calculate the exponent based on the RBF logic
    if simple:
        scaled_diff = -np.square(diff_warped) / (np.exp(extent))
        exponent = np.sum(scaled_diff, axis=-1)
    else:
        A, B = extent
        AAt = np.matmul(A, np.transpose(A))
        diag_exp_B = np.diag(np.exp(B))  # Element-wise exponential
        scale = AAt + diag_exp_B
        # Efficiently compute the quadratic form
        inv_scale = np.linalg.inv(scale)
        exponent = -0.5 * np.einsum('...i,...ij,...j->...', diff_warped, inv_scale, diff_warped)
    
    return np.exp(exponent)

## RBF kernel
def rbf_kernel(grid_coords, center, extent, simple = True):
    """This function computes the RBF kernel values for a given set of grid coordinates.

    Args:
        grid_coords (np.ndarray): Coordinates of the grid points, shape (..., 2).
        center (np.ndarray): Center of the RBF kernel, shape (2,).
        extent: Parameter controlling the Gaussian's extent. If simple=True, a scalar or array-like of shape (2,); else, a tuple (A, B).
        simple (bool): If True, use a diagonal covariance matrix. If False, use a full covariance matrix.
    Returns:
        np.ndarray: RBF kernel values, shape (...).
    """

    diff = grid_coords - center
    if simple:
        scaled_diff = -np.square(diff) / (np.exp(extent))
        exponent = np.sum(scaled_diff, axis=-1)
    else:
        A,B = extent
        AAt = np.matmul(A, np.transpose(A))
        diag_exp_B = np.diag(np.exp(B))  # Apply exponential element-wise
        scale = AAt + diag_exp_B
        exponent = -0.5 * np.einsum('...ik,...kl,...il->...i', diff, np.linalg.inv(scale), diff)


    return np.exp(exponent)

def create_blob(center, extent, shape, simple = True):
    """ 
    This function creates individual spatial factor
    Args:
        center (array-like): Center of the blob, shape (2,).
        extent: Parameter controlling the Gaussian's extent. If simple=True, a scalar or array-like of shape (2,); else, a tuple (A, B).
        shape (tuple): Shape of the output array.
        simple (bool): If True, use a diagonal covariance matrix. If False, use a full covariance matrix.
    Returns:   
        np.ndarray: Blob values, shape (shape[0], shape[1]).
    """

    if center is None:
        return np.zeros(shape)
    # Create a grid of coordinates
    x_coords = np.arange(shape[0])
    y_coords = np.arange(shape[1])
    grid_coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    # Calculate the RBF kernel values for the entire grid
    blob = rbf_kernel(grid_coords, center, extent, simple).reshape(shape)

    # # Temporary
    # warp_amp = np.random.uniform(low=2, high=4)
    # warp_freq = np.random.uniform(low=0.1, high=0.3)
    # blob = irregular_kernel(grid_coords, center, extent, warp_amp, warp_freq, simple).reshape(shape)

    return blob

def create_spatial_factor(node_placement, nx, ny, extent, num_nodes, simple=True):
    """ This function creates the spatial factors for the grid.
    Args:
        node_placement (list): List of node placements, each with shape (2,).
        nx (int): Number of rows in the grid.
        ny (int): Number of columns in the
        extent: Parameter controlling the Gaussian's extent. If simple=True, a scalar or array-like of shape (2,); else, a tuple (A, B).
        num_nodes (int): Number of nodes in the grid.
        simple (bool): If True, use a diagonal covariance matrix. If False, use a full covariance matrix.
    Returns:
        np.ndarray: Spatial factors, shape (num_nodes, nx * ny).

    """

    if simple:
        extents = np.random.uniform(low=extent["low"], high=extent["high"], size=(num_nodes,2))
        spatial_factor = np.array([create_blob(node_placement[k], extents[k], (ny,nx), simple) for k in range(len(node_placement))])
    else:
        ratio = 0.75
        extent_low, extent_high = extent["low"] * ratio, extent["high"] * ratio
        A = np.random.uniform(low=-5, high=5, size=(num_nodes,2,2))
        B = np.random.uniform(low=extent_low, high=extent_high, size=(num_nodes,2))
        spatial_factor = np.array([create_blob(node_placement[k], (A[k],B[k]), (ny,nx), simple) for k in range(len(node_placement))])

    return spatial_factor.reshape(num_nodes, -1)

def generate_grid_noise(cfg, num_samples):
    """ This function generates grid noise based on the specified noise type and scale.
    Args:
        cfg: Configuration object containing noise parameters.
        num_samples (int): Number of samples to generate.
    Returns:
        np.ndarray: Generated grid noise, shape (num_samples, num_variates, time_length, ny * nx).
    """

    noise_type = cfg.grid_noise
    noise_scale = cfg.grid_noise_scale
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_scale, (num_samples, cfg.num_variates, cfg.time_length, cfg.ny*cfg.nx))
    elif noise_type == "none":
        noise = 0
    else:
        raise ValueError("Noise type not recognized")
    return noise

def apply_spatial_factor(cfg, z_latent, spatial_factors, num_samples, map_type = 'linear'):
    """ This function applies the spatial factors to the latent variable to generate the grid.
    Args:
        cfg: Configuration object containing parameters.
        z_latent (np.ndarray): Latent variable, shape (num_samples, num_variates, time_length).
        spatial_factors (np.ndarray): Spatial factors, shape (num_variates, num_nodes, nx * ny).
        num_samples (int): Number of samples to generate.
        map_type (str): Type of mapping to apply ('linear', 'mlp').
    Returns:
        np.ndarray: Generated grid, shape (num_samples, num_variates, time_length, ny * nx).
    """

    if map_type == 'linear':
        X = np.einsum('btn,vnl->bvtl', z_latent, spatial_factors)
        X += generate_grid_noise(cfg, num_samples)  
    elif map_type == 'sigmoid': # this is not used in the paper
        X = np.einsum('btn,vnl->bvtl', z_latent, spatial_factors)
        a = np.random.uniform(2, 4)
        b = 0.5 * a
        # Apply the nonlinear transformation
        sigmoid_transformation = lambda x: a * (sigmoid(x)) - b
        X = sigmoid_transformation(X)
        X += generate_grid_noise(cfg, num_samples) 
    elif map_type == 'mlp':
        perturbation = np.random.uniform(1e-2, 5e-2)
        NN_spatial_transformer = SpatialDecoderNN(nx=cfg.nx,
                                                ny=cfg.ny,
                                                num_variates=cfg.num_variates,
                                                embedding_dim=1,
                                                lag=cfg.lag,
                                                num_nodes=cfg.num_nodes,
                                                scale=perturbation).cuda()
        Z = torch.tensor(z_latent, dtype=torch.float32).cuda()
        F = torch.tensor(spatial_factors, dtype=torch.float32).cuda()
        X = []
        # Z = Z.view(-1, cfg.num_nodes).cuda()
        # X = NN_spatial_transformer(Z, F).transpose(0,1)
        for i in range(Z.shape[0]):
            X.append(NN_spatial_transformer(Z[i], F).transpose(0,1).detach().cpu())

        X = torch.stack(X).numpy()
        X += generate_grid_noise(cfg, num_samples)  

    return X



def generate_grid(cfg, num_samples, node_placement, graph, func_list, noise_func_list, burnin_steps):
    """ This function generates the grid based on the specified parameters.
    Args:
        cfg: Configuration object containing parameters.
        num_samples (int): Number of samples to generate.
        node_placement (list): List of node placements, each with shape (2,).
        graph: Graph object containing the graph structure.
        func_list: List of functions to apply to the graph.
        noise_func_list: List of noise functions to apply to the graph.
        burnin_steps (int): Number of burn-in steps for the latent model.
    Returns:
        tuple: Generated grid (X), latent variable (z_latent), and spatial factors.
    """
    spatial_factors = []
    extent = {"low":cfg.node_extent_low, 
              "high":cfg.node_extent_high}
    # loop over all variates
    for i, group in enumerate(node_placement):
        # curr_node_placement = [node for node in node_placement[i] if node is not None]
        # curr_num_nodes = len(curr_node_placement)

        spatial_factors.append(create_spatial_factor(node_placement=node_placement[i],
                                               nx=cfg.nx,
                                               ny=cfg.ny,
                                               extent=extent,
                                               num_nodes=cfg.num_nodes,
                                               simple = False))

    spatial_factors = np.array(spatial_factors)

    latent_model = LATENT(num_samples=num_samples,
                        lag=cfg.lag,
                        time_length=cfg.time_length,
                        num_nodes=cfg.num_nodes,
                        graph=graph,
                        func_list=func_list,
                        noise_func_list=noise_func_list,
                        burnin_steps=burnin_steps,
                        base_noise_type=cfg.base_noise_type,
                        noise_scale=cfg.noise_scale,
                        history_dep_noise=cfg.hist_dep_noise)

    z_latent = latent_model.generate_data()

    X = apply_spatial_factor(cfg, z_latent, spatial_factors, num_samples, map_type = cfg.map_type)  
    return X, z_latent, spatial_factors


##########################################################################
#SAVAR: this is not used directly in the paper (used in Varimax)
##########################################################################

from src.utils.data_generation.SAVAR import SAVAR
def create_random_grid(node_placement, node_size, nx, ny, num_nodes, random=False):
    noise_weights = np.zeros((num_nodes, ny, nx))
    i = 0
    for i in range(len(node_placement)):
        if node_placement[i] is None:
            continue
        center = node_placement[i]
        y, x = center[0], center[1]
        noise_weights[i, y:y+node_size, x:x +
                      node_size] = create_random_mode((node_size, node_size), random=random)
    return noise_weights

def create_random_mode(size: tuple,
                       mu: tuple = (0, 0),
                       var: tuple = (.5, .5),
                       position: tuple = (3, 3, 3, 3),
                       Sigma: np.ndarray = None,
                       random: bool = True) -> np.ndarray:
    """
    Creates a positive-semidefinite matrix to be used as a covariance matrix of two var
    Then use that covariance to compute a pdf of a bivariate gaussian distribution which
    is used as mode weight. It is random but enfoced to be spred.
    Inspired in:  https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    and https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices

    :param random: Does not create a random, insted uses a ind cov matrix
    :param size
    :param mu tuple with the x and y mean
    :param var used to enforce spread modes. (0, 0) = totally random
    :param position: tuple of the position of the mean
    :param plot:
    """

    # Unpack variables
    size_x, size_y = size
    x_a, x_b, y_a, y_b = position
    mu_x, mu_y = mu
    var_x, var_y = var

    # In case of non invertible
    if Sigma is not None:
        Sigma_o = Sigma.copy()
    else:
        Sigma_o = Sigma

    # Compute the position of the mean
    X = np.linspace(-x_a, x_b, size_x)
    Y = np.linspace(-y_a, y_b, size_y)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Mean vector
    mu = np.array([mu_x, mu_y])

    # Compute almost-random covariance matrix
    if random:
        Sigma = np.random.rand(2, 2)
        Sigma = np.dot(Sigma, Sigma.transpose())  # Make it invertible
        Sigma += np.array([[var_x, 0], [0, var_y]])
    else:
        if Sigma is None:
            Sigma = np.asarray([[0.5, 0], [0, 0.5]])

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    # The actual weight
    Z = np.exp(-fac / 2) / N

    if not np.isfinite(Z).all() or (Z > 0.5).any():
        Z = create_random_mode(size=size, mu=mu, var=var, position=position,
                               Sigma=Sigma_o, random=random)

    return Z

def compute_mapping(node_size, node_placement, nx, ny):
    """Given node placements, calculate the  spatial mapping.
    This is done based on the closest distance of each grid point to the nodes

    Args:
        node_size (float): size of each node
        node_placement (List[List]): where each node is present
        nx (int): Number of rows
        ny (int): Number of columns
    Return:
        spatial_mapping (np.ndarray): Shape [num_variates, num_grid_points] characterizing the
                                      assignment of each grid point to the appropriate node.
    """
    # replace None with [inf, inf]
    node_placement = [[[float('inf'), float(
        'inf')] if x is None else x for x in r] for r in node_placement]
    # Adjust node placement to center
    node_placement = np.array(node_placement) + node_size/2

    # note that the convention is reversed here
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx))
    X = X[np.newaxis, :].transpose((0, 2, 1))
    Y = Y[np.newaxis, :].transpose((0, 2, 1))

    node_placement = node_placement[:, :, np.newaxis, np.newaxis]
    grid = np.concatenate((Y[..., np.newaxis], X[..., np.newaxis]), axis=-1)

    # Calculate distances using broadcasting
    # Calculate mapping based on closest node
    distances = np.sum(np.square(grid-node_placement), axis=-1)

    spatial_mapping = np.argmin(distances, axis=1).reshape(-1, nx*ny)

    return spatial_mapping

def generate_savar(cfg, num_samples, node_placement, graph, func_list, noise_func_list, burnin_steps):

    mode_weights = []
    # loop over all variates
    for i, group in enumerate(node_placement):
        mode_weights.append(create_random_grid(node_placement=node_placement[i],
                                               node_size=cfg.node_dist,
                                               nx=cfg.nx,
                                               ny=cfg.ny,
                                               num_nodes=cfg.num_nodes,
                                               random=cfg.random_node))

    mode_weights = np.array(mode_weights)

    savar_model = SAVAR(num_samples=num_samples,
                        lag=cfg.lag,
                        time_length=cfg.time_length,
                        mode_weights=mode_weights,
                        graph=graph,
                        func_list=func_list,
                        noise_func_list=noise_func_list,
                        burnin_steps=burnin_steps,
                        noise_strength=cfg.noise_scale,
                        history_dep_noise=cfg.hist_dep_noise)

    mapping = compute_mapping(node_size=cfg.node_dist,
                              node_placement=node_placement,
                              nx=cfg.nx,
                              ny=cfg.ny)
    return savar_model.generate_data(), mode_weights, mapping