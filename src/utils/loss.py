# metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from cdt.metrics import SHD
from scipy.optimize import linear_sum_assignment

import numpy as np
import os
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
from sklearn.model_selection import ParameterGrid
from scipy.spatial.distance import cdist

# TODO: Clean up


def create_Xweight(X_full, normalization='min-max'):

    batch, time, ny, nx = X_full.shape
    X_abs = torch.abs(X_full) + 1e-6
    X_log = torch.log(X_abs)
    X_log_1 = X_log - X_log.min()
    X_sum = X_log_1.sum(dim=(0, 1))

    # X_sum = X_abs.sum(dim=(0, 1))
    X_sum = X_sum.squeeze(-1)

    if normalization == 'max':
        # Normalize by max value
        X_sum = X_sum / X_sum.max()
    elif normalization == 'norm':
        # Normalize by the Frobenius norm
        norm = torch.norm(X_sum, p='fro')
        X_sum = X_sum / norm
    elif normalization == 'time':
        # Normalize by time
        X_sum = X_sum / (time*batch)
    elif normalization == 'min-max':
        # Normalize by min-max
        # Compute the actual range and adjust it by the buffer factor
        # buffer_factor = 0.1
        # X_range = X_sum.max() - X_sum.min()
        # min_soft = X_sum.min() - X_range * buffer_factor
        # max_soft = X_sum.max() + X_range * buffer_factor

        # Apply the soft min-max normalization
        X_sum = (X_sum - X_sum.min()) / (X_sum.max() - X_sum.min())
    X_sum_flattened = X_sum.view(-1)

    def plot_heatmap(X_sum):
        plt.figure(figsize=(8, 6))
        sns.heatmap(X_sum, cmap='hot', cbar=True)
        plt.title('Heatmap of X_sum')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('heatmap.png')
        plt.show()

    plot_heatmap(X_sum.detach().cpu().numpy())
    # average = torch.mean(X_sum_flattened)
    # X_sum_flattened[X_sum_flattened < average] = average
    return X_sum_flattened


def dynamic_peaks(X_weight, ny, nx, W):
    num_nodes, grid_size = W.shape
    WX = W * X_weight

    # Get indices of the maximum values in the flattened dimension
    max_indices_flat = torch.argmax(WX, dim=1)

    # Convert flat indices back to 2D indices using floor division and modulus for row and column indices
    rows = max_indices_flat // nx
    cols = max_indices_flat % nx

    # Stack the 2D indices along a new dimension and transpose to get the desired shape
    selected_indices = torch.stack((rows, cols), dim=1)

    return selected_indices


def create_laplacian(ny, nx):
    """
    Create the Laplacian matrix for a 2D grid of size ny by nx, following the new convention.
    Each grid point is connected to its immediate neighbors.
    """
    N = nx * ny
    L = torch.zeros(N, N)

    for y in range(ny):
        for x in range(nx):
            index = y * nx + x  # Adjusted index calculation
            # Connect to immediate neighbors
            if y > 0:
                L[index, index - nx] = -1  # Above
            if y < ny - 1:
                L[index, index + nx] = -1  # Below
            if x > 0:
                L[index, index - 1] = -1   # Left
            if x < nx - 1:
                L[index, index + 1] = -1   # Right

            # Connect to diagonal neighbors
            if y > 0 and x > 0:
                L[index, index - nx - 1] = -1  # Top-left
            if y > 0 and x < nx - 1:
                L[index, index - nx + 1] = -1  # Top-right
            if y < ny - 1 and x > 0:
                L[index, index + nx - 1] = -1  # Bottom-left
            if y < ny - 1 and x < nx - 1:
                L[index, index + nx + 1] = -1  # Bottom-right

            # Set the diagonal element
            L[index, index] = -L[index].sum()

    return L


def laplacian_regularization(W, laplacian, nx, ny):
    """
    Compute the Laplacian regularization term.
    W is the assignment matrix (num_nodes, nx*ny).
    """
    l = 1/(nx * ny)
    W = W.float()
    L = laplacian
    reg_term = l*torch.trace(torch.matmul(torch.matmul(W, L), W.T))
    return reg_term


def create_grid(nx, ny, mode='Euclidean', long_bounds=(-180, 180), lat_bounds=(-90, 90)):
    """
    Create a grid of size nx by ny. Supports regular indexing or geographic coordinates.

    Args:
    nx (int): Number of horizontal grid cells (width).
    ny (int): Number of vertical grid cells (height).
    mode (str, optional): Type of grid to create. 'Euclidean' for regular indexing, 'Geo' for geographic coordinates.
    long_bounds (tuple, optional): A tuple (min_longitude, max_longitude) defining the longitude bounds.
    lat_bounds (tuple, optional): A tuple (min_latitude, max_latitude) defining the latitude bounds.

    Returns:
    torch.Tensor: A 2D tensor where each row is a coordinate pair (y, x) or (latitude, longitude).
    """
    if mode == 'Euclidean':
        # Regular grid indexing
        h_coords = torch.arange(nx)
        v_coords = torch.arange(ny)
        # Switch in order and indexing
        Y, X = torch.meshgrid(v_coords, h_coords, indexing='ij')
        grid = torch.stack((Y.flatten()/ny, X.flatten()/nx), dim=-1)
        return grid.float()
    elif mode == 'Geo':
        if long_bounds is None or lat_bounds is None:
            raise ValueError(
                "Longitude and latitude bounds must be specified for geographic mode.")
        # Create linearly spaced coordinates within the provided bounds
        longitudes = torch.linspace(long_bounds[0], long_bounds[1], nx)
        latitudes = torch.linspace(lat_bounds[0], lat_bounds[1], ny)
        # Note: lat comes before long to keep (y, x) order
        Longitude, Latitude = torch.meshgrid(
            latitudes, longitudes, indexing='ij')
        geo_grid = torch.stack(
            (Longitude.flatten(), Latitude.flatten()), dim=-1)
        return geo_grid.float()

# Helper function to convert normalized [0,1] lat/lon to the correct ranges
def to_lat_long_radians(coords):
    lat = coords[..., 0] * torch.pi - torch.pi/2          # Latitude: [0,1] -> [0, pi]
    lon = coords[..., 1] * 2 * torch.pi - torch.pi   # Longitude: [0,1] -> [0, 2pi]
    return torch.stack((lat, lon), dim=-1)

# Haversine distance calculation assuming Earth's radius = 1
def haversine_distance(grid_coords, centers):
    # Convert normalized [0,1] latitude and longitude to correct ranges in radians
    grid_coords_rad = to_lat_long_radians(grid_coords)
    centers_rad = to_lat_long_radians(centers)
    
    dlat = grid_coords_rad[..., 0] - centers_rad[..., 0]
    dlon = grid_coords_rad[..., 1] - centers_rad[..., 1]
    

    # dlon = (dlon) % (torch.pi)
    # Apply the Haversine formula
    
    temp = torch.sin(dlat/2)**2 + torch.cos(centers_rad[..., 0]) * torch.cos(grid_coords_rad[..., 0]) * torch.sin(dlon/2)**2
    sqrt_temp = torch.sqrt(temp + 1e-6)
    sqrt_temp = torch.clamp(sqrt_temp, -1 + 1e-5, 1 - 1e-5)
    dist = 2 * torch.asin(sqrt_temp)
    # dist = 2 * torch.atan2(torch.sqrt(temp), torch.sqrt(1-temp)) 

    dist = dist.unsqueeze(-1)  
    return dist

def euclidean_distance(grid_coords, centers):
    grid_diff = grid_coords - centers 
    return grid_diff
    

def calculate_distance(grid_coords, centers, distance_mode='Euclidean'):
    """
    Calculate the distance between grid coordinates and centers in different ways (Euclidean, Haversine).

    Args:
        grid_coords (Tensor): Tensor of shape (1, n_centers, ny*nx, 2), normalized [0,1].
                              For Haversine, [0,1] corresponds to latitude [0,pi] and longitude [0,2pi].
        centers (Tensor): Tensor of shape (1, n_centers, 1, 2), normalized [0,1].
                          For Haversine, [0,1] corresponds to latitude [0,pi] and longitude [0,2pi].
        distance_mode (str): The mode for distance calculation ('Euclidean', 'Haversine').

    Returns:
        Tensor: Distance between each grid coordinate and center.
                Shape (1, n_centers, ny*nx, 1)
    """
    
    if distance_mode == 'Euclidean':
        return euclidean_distance(grid_coords, centers)
    elif distance_mode == 'Haversine':
        return haversine_distance(grid_coords, centers)
    else:
        raise ValueError(f"Unknown distance_mode: {distance_mode}")
    

# def calculate_distance(grid_coords, centers, distance_mode = 'Euclidean'):
#     """
#     Calculate the distance between grid coordinates and centers in different ways (Cartesian, Haversine).

#     Args:
#         grid_coords (Tensor): Tensor of shape (1, n_centers, ny*nx, 2), normalized [0,1] (Haversine: longitude and latitude).
#         centers (Tensor): Tensor of shape (1, n_centers, 1, 2), normalized [0,1] (Haversine: longitude and latitude).

#     Returns:
#         Tensor: Distance between each grid coordinate and center.
#                 Shape (1, n_centers, ny*nx, 1)
#     """

#     # Ensure input tensors are floating point
#     grid_coords = grid_coords.float()
#     centers = centers.float()

#     if distance_mode == 'Euclidean':
#         return grid_coords - centers
    
#     # Extract longitude and latitude from grid_coords and centers
#     # Shape: (1, n_centers, ny*nx, 2) -> (1, n_centers, ny*nx) for longitude and latitude
#     lon_grid = grid_coords[..., 1] * 2 * math.pi  # [0, 2π]
#     lat_grid = grid_coords[..., 0] * math.pi      # [0, π]

#     lon_centers = centers[..., 1] * 2 * math.pi   # [0, 2π]
#     lat_centers = centers[..., 0] * math.pi       # [0, π]

#     # Calculate delta_lon and delta_lat
#     delta_lon = lon_grid - lon_centers            # Shape: (1, n_centers, ny*nx)
#     delta_lat = lat_grid - lat_centers            # Shape: (1, n_centers, ny*nx)

#     # Apply approximate haversine formula for longitude (Δλ)
#     # Calculate the mean latitude for the cosine term
    
#     # mean_lat = (lat_grid + lat_centers) / 2       # Shape: (1, n_centers, ny*nx)

#     delta_x = torch.cos(lat_grid) * delta_lon     # Corrected Δx using cos(mean latitude)

#     # # Apply Exact haversine formula for longitude (Δλ)
#     # delta_lambda = delta_lon / 2
#     # sin_delta_lambda = torch.sin(delta_lambda) ** 2

#     # mean_lat = (lat_grid + lat_centers) / 2       # Shape: (1, n_centers, ny*nx)
#     # cos_mean_lat = torch.cos(mean_lat)

#     # # Haversine formula for longitude correction
#     # delta_x = 2 * torch.arcsin(torch.sqrt(cos_mean_lat * sin_delta_lambda))  # Corrected Δx using haversine formula

#     delta_y = delta_lat                           # Δy is simply the latitude difference

#     # Stack the results to match the output shape (1, n_centers, ny*nx, 2)
#     distances = torch.stack([delta_y, delta_x], dim=-1)  # Shape: (1, n_centers, ny*nx, 2)

#     return distances/math.pi  # Shape: (1, n_centers, ny*nx, 2)


def haversine_mean(coords, weights):
    """
    Calculate the weighted mean of geographic coordinates on a sphere.

    Args:
    coords (torch.Tensor): Coordinate tensor of shape (2, N) where the first row is latitudes and the second is longitudes.
    weights (torch.Tensor): Weights for each coordinate.

    Returns:
    torch.Tensor: Weighted mean latitude and longitude.
    """
    # Convert degrees to radians
    lat_rad = coords[0, :] * torch.pi / 180
    lon_rad = coords[1, :] * torch.pi / 180

    # Compute weighted mean of sines and cosines of coordinates
    sum_of_weights = torch.sum(weights)
    avg_sin_lat = torch.sum(torch.sin(lat_rad) * weights) / sum_of_weights
    avg_cos_lat = torch.sum(torch.cos(lat_rad) * weights) / sum_of_weights
    avg_sin_lon = torch.sum(torch.sin(lon_rad) * weights) / sum_of_weights
    avg_cos_lon = torch.sum(torch.cos(lon_rad) * weights) / sum_of_weights

    # Convert back to angles
    mean_lat = torch.atan2(avg_sin_lat, avg_cos_lat) * 180 / torch.pi
    mean_lon = torch.atan2(avg_sin_lon, avg_cos_lon) * 180 / torch.pi

    return torch.tensor([mean_lat, mean_lon])


def estimate_center(W, nx, ny, X_weight, grid_coords, mode='Euclidean'):
    num_nodes, grid_size = W.shape

    coords = grid_coords.T * X_weight
    if mode == 'Euclidean':
        centers = torch.matmul(W, coords.T.float())
        weighted_num_assignments = W@X_weight
        centers = centers / weighted_num_assignments.unsqueeze(1)
    elif mode == 'Geo':
        centers = torch.zeros((num_nodes, 2))
        weighted_num_assignments = W @ X_weight

        for i in range(num_nodes):
            node_weights = W[i, :] * X_weight
            centers[i, :] = haversine_mean(coords, node_weights)
    return centers


# def haversine_distance(lat1, lon1, lat2, lon2):
#     """
#     Calculate the Haversine distance between two points on the earth.
#     Input and output are in degrees for latitudes and longitudes.

#     Args:
#     lat1, lon1: Latitudes and longitudes of the first point or points.
#     lat2, lon2: Latitudes and longitudes of the second point or points.

#     Returns:
#     Torch tensor: Distance between the points in kilometers.
#     """
#     # Earth radius in kilometers
#     R = 6371.0

#     # Convert degrees to radians
#     lat1, lon1, lat2, lon2 = [(x * torch.pi / 180).cuda()
#                               for x in (lat1, lon1, lat2, lon2)]

#     # Haversine formula
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = torch.sin(dlat/2)**2 + torch.cos(lat1) * \
#         torch.cos(lat2) * torch.sin(dlon/2)**2
#     c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

#     return R * c  # Distance in kilometers


def spatial_prior(W, nx, ny, X_weight, grid_coords, intracluster_factor=100.0, intercenter_factor=1.0, cluster_factor=1.0):
    num_nodes, grid_size = W.shape
    coords = grid_coords

    # peaks = dynamic_peaks(X_weight, ny, nx, W)
    centers = estimate_center(W, nx, ny, X_weight, grid_coords)
    # centers = peaks

    # Broadcasting the coordinates and centers for vectorized distance calculation
    # Add an extra dimension to coords so it can be broadcasted against centers
    expanded_coords = coords.unsqueeze(1)
    # Add an extra dimension to centers for broadcasting
    expanded_centers = centers.unsqueeze(0)

    # Calculate the intracluster distance
    distances = torch.sum((expanded_coords - expanded_centers) ** 2, dim=2)

    intra_dist = torch.mul(W, distances.T)
    # inter_dist = distances.T - intra_dist
    intracluster = torch.sum(intra_dist)
    # intercluster = torch.sum(inter_dist)
    l1 = intracluster_factor/grid_size
    # Calculate the intercluster distance
    differences = centers[:, None, :] - centers[None, :, :]
    intercenter = torch.sum(differences ** 2, axis=2).sum()
    # l2 = 1
    l2 = intercenter_factor/num_nodes

    # Calculate the prior
    prior = cluster_factor*((l1*intracluster) - (l2*intercenter))
    # cluster_prior_dict = {'intracluster': l1*intracluster,
    #                       'intercenter': l2*intercenter}
    
    return prior, centers


def spatial_prior_geo(W, nx, ny, X_weight, grid_coords, intracluster_factor=100.0, intercenter_factor=1.0, cluster_factor=1.0):
    num_nodes, grid_size = W.shape
    coords = grid_coords

    centers = estimate_center(W, nx, ny, X_weight, grid_coords, mode='Geo')

    # Broadcasting the coordinates and centers for vectorized distance calculation
    expanded_coords_lat = coords[0].unsqueeze(1)  # Latitudes
    expanded_coords_lon = coords[1].unsqueeze(1)  # Longitudes
    expanded_centers_lat = centers[:, 0].unsqueeze(0)
    expanded_centers_lon = centers[:, 1].unsqueeze(0)

    # Calculate the intracluster distance using haversine
    distances = haversine_distance(
        expanded_coords_lat, expanded_coords_lon,
        expanded_centers_lat, expanded_centers_lon
    )
    distances = torch.sum(distances, dim=0)
    intra_dist = torch.mul(W, distances.unsqueeze(0).T)
    intracluster = torch.sum(intra_dist)
    l1 = intracluster_factor / grid_size

    # Calculate the intercluster distance using haversine
    intercenter = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            intercenter[i, j] = haversine_distance(
                centers[i, 0], centers[i, 1],
                centers[j, 0], centers[j, 1]
            )
    intercenter = intercenter.sum()
    l2 = intercenter_factor / num_nodes

    # Calculate the prior
    prior = cluster_factor * ((l1 * intracluster) - (l2 * intercenter))

    return prior, centers


def mean_normalize(X):
    mean = torch.mean(X, dim=0)
    X_normalized = X - mean
    return X_normalized


def std_normalize(X):
    std = torch.std(X, axis=0)
    return X/std


def permute_peaks(centers, peaks):
    # Assuming centers and peaks are PyTorch tensors, convert them to numpy arrays
    centers = centers.cpu().numpy()
    peaks = peaks.cpu().numpy()

    # Calculate the distance matrix between each center and peak
    distance_matrix = np.linalg.norm(
        centers[:, None, :] - peaks[None, :, :], axis=2)

    # Solve the assignment problem
    center_indices, peak_indices = linear_sum_assignment(distance_matrix)

    # Reorder peaks according to the assignment to minimize overall distance
    optimal_peaks = peaks[peak_indices]
    return optimal_peaks


def correlation_prior(W, X_full, X_weight, grid_coords, correlation_factor=10.0):
    num_nodes, grid_size = W.shape
    batch, time, ny, nx = X_full.shape
    X = mean_normalize(X_full[0])
    # centers = estimate_center(W, nx, ny, X_weight, grid_coords)
    peaks = dynamic_peaks(X_weight, ny, nx, W)
    # permuted_peaks = permute_peaks(centers, peaks)
    # np.save('visualization/peaks.npy', peaks.detach().cpu().numpy())
    X_peaks = X[:, peaks[:, 0].long(), peaks[:, 1].long()]
    X_peaks = X_peaks.squeeze(-1)

    # X_agg = torch.einsum("tdl,ln->tdn", X.view(time, ny*nx, 1).permute(0,2,1), W.T)
    # X_agg = X_agg.permute(0,2,1).squeeze(-1)/torch.sum(W, axis = 1)

    X = X.squeeze(-1)
    X = X.view(time, -1)
    correlations = torch.matmul(X.T, X_peaks)
    correlations = correlations / time
    intra_correlations = correlations.T*W
    intra_correlation_nodes = torch.sum(intra_correlations, axis=1)
    intra_correlation_sum = intra_correlation_nodes.sum()
    # intra_correlations_difference = intra_correlation_nodes[:,None] - intra_correlation_nodes[None,:]
    # intra_correlations_difference_sum = torch.sum(intra_correlations_difference ** 2)

    W_conj = torch.abs(1 - W)
    inter_correlations = correlations.T*W_conj
    inter_correlation_sum = torch.sum(
        inter_correlations, axis=1).sum() / (num_nodes - 1)

    # differences = peaks[:, None, :] - peaks[None, :, :]
    # interpeaks = torch.sum(differences ** 2, axis=2).sum() / (num_nodes * correlation_factor)
    return correlation_factor*(inter_correlation_sum - intra_correlation_sum)
