import sys
import os
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
# from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
# import community_detection
# import community_detection_synthetic
import time
import utils_linear_response
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import os


# def community_detect(data, heuristic = False):
#     climate_variable = 'tos'
#     lon_variable = 'lon'
#     lat_variable = 'lat'
#     rand_sample_k = 1000000
#     rand_sample_eta = 1000000
#     q_k = 0.95
#     if heuristic:
#         q_eta = 0.15
#     else:
#         q_eta = 1 # ---> no heuristic


#     start = time.time()

#     community_map, single_communities, average_signals, cumulative_signals = community_detection_synthetic.community_detection(data,climate_variable,lon_variable,lat_variable,
#                                                                                                                     rand_sample_k,rand_sample_eta,q_k,q_eta)
#     end = time.time()

#     print('Finished in '+str(round(end - start, 2))+' seconds')
#     return community_map, single_communities, average_signals, cumulative_signals

def view_community(longitudes, latitudes, community_map_noH, community_map_H, figure_dir):
    fig = plt.figure(figsize=(14,11))

    ax = fig.add_subplot(211)  
        
    map = Basemap(projection='cyl',
                llcrnrlat=-60,urcrnrlat=60,\
                llcrnrlon=0,urcrnrlon=360)

    map.drawcoastlines()
    map.drawparallels(np.arange(-60.,75,30),  labels=[1,0,0,0], fontsize = 28,linewidth=0.001)
    #map.drawmeridians(np.arange(0.,360.,90.), labels=[0,0,0,1], fontsize = 28,linewidth=0.001)
    map.fillcontinents(color = 'black')

    map.pcolor(longitudes,latitudes,community_map_noH,cmap=plt.cm.turbo)
    #map.pcolor(longitudes+100,latitudes,community_map_noH,cmap=plt.cm.turbo)
    #cb=plt.colorbar(location='bottom',aspect=20,pad=0.08)
    #cb.ax.tick_params(labelsize=37)
    plt.title('Communities', fontsize = 37)
    plt.title('(a)', loc = 'left', fontsize = 37)

    ax = fig.add_subplot(212)  
        
    map = Basemap(projection='cyl',
                llcrnrlat=-60,urcrnrlat=60,\
                llcrnrlon=0,urcrnrlon=360)

    map.drawcoastlines()
    map.drawparallels(np.arange(-60.,75,30),  labels=[1,0,0,0], fontsize = 28,linewidth=0.001)
    map.drawmeridians(np.arange(0.,360.,90.), labels=[0,0,0,1], fontsize = 28,linewidth=0.001)
    map.fillcontinents(color = 'black')

    map.pcolor(longitudes,latitudes,community_map_H,cmap=plt.cm.turbo)


    # Plot domain id
    plt.title('Communities (Spatially contiguous)', fontsize = 37)
    plt.title('(b)', loc = 'left', fontsize = 37)

    fig.savefig(f'{figure_dir}/communities_local_vs_nonlocal.pdf',bbox_inches='tight') # bbox_inches='tight'

def causality_linear_response(average_signals):
    signals = average_signals
        # Time length
    n_time = np.shape(signals)[1]
    # Number of time series
    n_ts = np.shape(signals)[0]
    # inputs
    tau_max = 10 * 10  # ~12 years in monthly resolution
    standardized = 'yes' # response computed via correlations functions
    response_matrix_filter_10yrs = utils_linear_response.response(signals,tau_max,standardized)

    ############################## Step (a)
    # Compute the lag 1 autocorrelation

    # lag-1 autocorrelation
    print('Computing lag-1 autocorr')
    phi = utils_linear_response.phi_vector(signals)

    ############################## Step (b)
    # Compute standard deviations of each time series

    # sigmas
    print('Computing sigmas')
    sigmas = utils_linear_response.sigmas(signals)

    ### Parameters

    # we compute responses up to a lag tau_max
    tau_max = 100
    # we compute correlations
    standardized = 'yes'

    s = 8
    # This correspondes to +/- 5 sigmas
    s_minus, s_plus = utils_linear_response.compute_quantile_analytical_tau_discrete(signals,phi,sigmas,tau_max,s,standardized='yes')
    return response_matrix_filter_10yrs, s_minus, s_plus

# Function to define node_strength with statistical significance
# (There are better ways to write this.)

def node_strength_significance(response_matrix,conf_bounds_plus,conf_bounds_minus, absolute_value):
    
    # Inputs
    # - response_matrix
    # - significance_right_tail: for example the 99th percentile of the ensemble of null models
    # - significance_left_tail: for example the 1st percentile of the ensemble of null models
    
    # Outputs:
    # - strengths_j_k: strength of the connection j -> k
    # If the original response matrix is n by n, strengths_j_k will be n x (n - 1)
    # as it will not consider self links
    
    time = np.shape(response_matrix)[0]
    # number of rows = number of columns = n
    n = np.shape(response_matrix)[1]
    
    # response_matrix_significant: assign zero if not significant
    response_matrix_significant = response_matrix.copy()
    
    # if you are not significant we change you to zero
    indices = (response_matrix < conf_bounds_plus) & (response_matrix > conf_bounds_minus)
    response_matrix_significant[indices] = 0
    
    # Strength of link j -> k
    strengths_j_k = np.zeros([n,n])
    
    if absolute_value == 'yes':
    
        # Response j -> k in absolute value
        abs_response_j_k = np.abs(response_matrix_significant[1:])
        # Compute strength of j -> k
        strengths_j_k = np.transpose(np.sum(abs_response_j_k,axis = 0))
    
        # When computing strengths we remove the j -> j connection
        # remove diagonal
        strengths_j_k_off_diagonal = strengths_j_k[~np.eye(strengths_j_k.shape[0],dtype=bool)].reshape(strengths_j_k.shape[0],-1)
    
        # Strength of node j
        strengths_j = np.sum(strengths_j_k_off_diagonal,axis = 1)
        
    elif absolute_value == 'no':
    
        # Response j -> k in absolute value
        abs_response_j_k = response_matrix_significant[1:]
        # Compute strength of j -> k
        strengths_j_k = np.transpose(np.sum(abs_response_j_k,axis = 0))
    
        # When computing strengths we remove the j -> j connection
        # remove diagonal
        strengths_j_k_off_diagonal = strengths_j_k[~np.eye(strengths_j_k.shape[0],dtype=bool)].reshape(strengths_j_k.shape[0],-1)
    
        # Strength of node j
        strengths_j = np.sum(strengths_j_k_off_diagonal,axis = 1)    
    
    return strengths_j_k, strengths_j

def compute_linkmap(data, strength_map_tos, response_matrix, s_plus, s_minus, longitudes, latitudes):
    absolute_value = 'no'
    strengths_j_k, strengths_j = node_strength_significance(response_matrix[0:5],s_plus[0:5],s_minus[0:5],absolute_value)
    # Print causal strength maps

    strength_ENSO = single_communities.copy()
    enso_index = 0

    strength_IO = single_communities.copy()
    io_index = 1

    strength_TA_S = single_communities.copy()
    ta_s_index = 2

    strength_TA_N = single_communities.copy()
    ta_n_index = 4


    for k in range(len(single_communities)): 
        # enso
        strength_ENSO[k] = single_communities[k] * strengths_j_k[enso_index,k]
        # io
        strength_IO[k] = single_communities[k] * strengths_j_k[io_index,k]
        # ta_s
        strength_TA_S[k] = single_communities[k] * strengths_j_k[ta_s_index,k]
        # ta_n
        strength_TA_N[k] = single_communities[k] * strengths_j_k[ta_n_index,k]
        
    # remove your self
    strength_ENSO = np.delete(strength_ENSO,enso_index,axis=0)
    strength_IO = np.delete(strength_IO,io_index,axis=0)
    strength_TA_S = np.delete(strength_TA_S,ta_s_index,axis=0)
    strength_TA_N = np.delete(strength_TA_N,ta_n_index,axis=0)
        
    strength_map_ENSO = np.nansum(strength_ENSO,axis = 0)
    strength_map_IO = np.nansum(strength_IO,axis = 0)
    strength_map_TA_S = np.nansum(strength_TA_S,axis = 0)
    strength_map_TA_N = np.nansum(strength_TA_N,axis = 0)
    #strength_map_ENSO[strength_map_ENSO == 0] = np.nan

    mask = data.copy()
    mask = np.std(data,axis = 0)
    mask[mask == 0.] = np.nan
    mask[~np.isnan(mask)] = 0
    strength_map_tos = strength_map_tos+mask
    strength_map_ENSO = strength_map_ENSO + mask
    strength_map_IO = strength_map_IO + mask
    strength_map_TA_S = strength_map_TA_S + mask
    strength_map_TA_N = strength_map_TA_N + mask

    return strength_map_ENSO, strength_map_IO, strength_map_TA_S, strength_map_TA_N

def construct_adjacency_matrix(response_matrix, null_response_high_percentile, tau_max):
    # Initialize the adjacency matrix with the same shape as response_matrix
    # and fill it with zeros
    time, num_nodes, _ = response_matrix.shape
    adjacency_matrix = np.zeros_like(response_matrix)
    
    # Iterate over the time slices up to tau_max (inclusive)
    for t in range(tau_max + 1):
        # Compare the matrices and set adjacency_matrix[t, i, j] to 1 where
        # the condition is met, otherwise it remains 0
        adjacency_matrix[t] = (response_matrix[t] > null_response_high_percentile[t]).astype(int)
    np.fill_diagonal(adjacency_matrix[0], 0)

    return adjacency_matrix[:tau_max+1]


def prune_matrix_by_sparsity(adjacency_matrix, num_node):
    # Compute the total number of connections for each node across all time steps
    total_connections = adjacency_matrix.sum(axis=0).sum(axis=0)
    
    # Find the indices of the top 'num_node' nodes
    # 'argsort' returns indices that would sort the array, and we take the last 'num_node' indices
    top_nodes_indices = np.argsort(total_connections)[:num_node]
    
    # Sort the indices to maintain the order of nodes as in original data
    top_nodes_indices.sort()
    
    # Create a new adjacency matrix including only the top nodes
    pruned_matrix = adjacency_matrix[:, top_nodes_indices, :][:, :, top_nodes_indices]
    
    return pruned_matrix,top_nodes_indices

def graph_permute(W, W_oh, predictions, top_nodes_indices):
    print("W", W.shape, "W_oh", W_oh.shape)
    num_nodes = W.shape[0]
    if W.shape != W_oh.shape:
        W_oh = W_oh[top_nodes_indices,:]
    assert W.shape == W_oh.shape
    W = W.argmax(axis=0).flatten()
    W_oh = W_oh.argmax(axis=0).flatten()

    cm = confusion_matrix(W_oh,W)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    print("ind", row_ind, col_ind)
    predictions = predictions[:, col_ind, :][:, :, col_ind]
    return predictions, cm[row_ind, col_ind].sum()/cm.sum()

def get_off_diagonal(A):
    # assumes A.shape: (batch, x, y)
    M = np.invert(np.eye(A.shape[1], dtype=bool))
    return A[M]

def evaluate_results(adj_matrix, predictions, aggregated_graph=False, is_last=False):
    assert adj_matrix.shape == predictions.shape, "Dimension of adj_matrix should match the predictions"
    adj_matrix[adj_matrix != 0] = 1
    predictions[predictions != 0] = 1

    preds = np.abs(np.round(predictions))
    truth = adj_matrix.flatten()

    # calculate shd
    m = adj_matrix.shape[0]
    f1_inst = f1_score(get_off_diagonal(adj_matrix[0]).flatten(), get_off_diagonal(predictions[0]).flatten())
    f1_lag = f1_score(adj_matrix[1:].flatten(), preds[1:].flatten())
    
    f1 = f1_score(truth, preds.flatten())
    
    preds = preds.flatten()
    zero_edge_accuracy = np.sum(np.logical_and(
        preds == 0, truth == 0))/np.sum(truth == 0)
    one_edge_accuracy = np.sum(np.logical_and(
        preds == 1, truth == 1))/np.sum(truth == 1)
    print("truth", truth)
    print("preds", preds)
    accuracy = accuracy_score(truth, preds)
    precision = precision_score(truth, preds)
    recall = recall_score(truth, preds)
    tnr = zero_edge_accuracy
    tpr = one_edge_accuracy

    print("Accuracy score:", accuracy)
    print("F1 score:", f1)
    print("Precision score:", precision)
    print("Recall score:", recall)
    print("Accuracy on '0' edges", tnr)
    print("Accuracy on '1' edges", tpr)
    # print("Cluster Accuracy:", cluster_acc)
     
    if not aggregated_graph:
        print("F1 inst", f1_inst)
        print("F1 lag", f1_lag)
    
    # also return a dictionary of metrics
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tnr': tnr,
        'tpr': tpr,
        # 'cluster_acc': cluster_acc
    }

    if not aggregated_graph:

        metrics['f1_inst'] = f1_inst
        metrics['f1_lag'] = f1_lag

    if is_last:
        metrics_last = {f'{key}_is_last': value for key, value in metrics.items()}
        return metrics_last
    return metrics

def convert_to_onehot(community_map):
    ny, nx = community_map.shape
    n = int(np.max(community_map))  # Assuming communities are labeled from 1 to n
    community_map_onehot = np.zeros((n, ny * nx), dtype=int)
    
    # Flatten the community_map to use 1D indexing
    community_map_flat = community_map.ravel() - 1  # Convert to 0-based index
    community_map_flat.astype(int)
    # Create a linear index for the community_map_flat
    linear_index = np.arange(ny * nx)
    
    # Use numpy advanced indexing to set the values
    # community_map_onehot[community_map_flat, linear_index] = 1
    for i in range(len(community_map_flat)):
        community_map_onehot[int(community_map_flat[i]), i] = 1
    return community_map_onehot

def compute_adj_grid(community_map_oh, adj):
    # Calculate the pseudo inverse of community_map_oh
    # Shape of community_map_oh is (n, ny*nx)
    community_map_oh_pinv = np.linalg.pinv(community_map_oh)

    # Initialize adj_grid with the appropriate shape (t, ny*nx, ny*nx)
    t = adj.shape[0]
    ny_nx = community_map_oh.shape[1]
    adj_grid = np.zeros((t, ny_nx, ny_nx))

    # Compute adj_grid for each time slice in adj
    for i in range(t):
        adj_grid[i] = community_map_oh_pinv @ adj[i] @ community_map_oh

    return adj_grid

