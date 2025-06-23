from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats
from scipy.stats import pearsonr
from copy import deepcopy
# Tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
# from tigramite.independence_tests import ParCorr
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import LinearMediation, Prediction
from .munkres import Munkres

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
    Args:
        x: data to be sorted
        y: target data
    Returns:
        corr_sort: correlation matrix between x and y (after sorting)
        sort_idx: sorting index
        x_sort: x after sorting
        method: correlation method ('Pearson' or 'Spearman')
    """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]
    return corr_sort, sort_idx, x_sort

def compute_mcc(mus_train, ys_train, correlation_fn):
    """Computes score based on both training and testing codes and factors."""
    score_dict = {}
    result = np.zeros(mus_train.shape)
    result[:ys_train.shape[0],:ys_train.shape[1]] = ys_train
    for i in range(len(mus_train) - len(ys_train)):
        result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
    corr_sorted, sort_idx, mu_sorted = correlation(mus_train, result, method=correlation_fn)
    mcc = np.mean(np.abs(np.diag(corr_sorted)[:len(ys_train)]))
    return mcc, torch.tensor(sort_idx, dtype=torch.int32)

def pcmci(mus, lag):
    pc_alpha = 0.2
    batch, num_node = mus.shape
    # mu = mus.reshape(batch, num_node)
    mu = mus
    ind_test = ParCorr(significance="analytic")
    pcmci_test = PCMCI(
            dataframe=pp.DataFrame(mu),  # Input data for PCMCI T times K
            cond_ind_test=ind_test,
            verbosity=False,
        )

    result = pcmci_test.get_lagged_dependencies(selected_links=None,
                                                tau_min=1,
                                                tau_max=lag,
                                                val_only=False)
    variance_vars = np.std(pcmci_test.dataframe.values[0], axis = 0)
    Phi = result['val_matrix']
    Phi[result['p_matrix'] > pc_alpha] = 0
    Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]
    phi = np.moveaxis(deepcopy(Phi), 2, 0)
    phi[np.abs(phi) > 0] = 1
    return phi

def get_permutation_and_eval_mapping(W, W_pred):
    # convert W and W_pred to numpy
    W_np = W.detach().cpu().numpy()
    W_pred_np = W_pred.detach().cpu().numpy()   
    num_variates = W_np.shape[0]

    # obtain optimal matching 
    cm = confusion_matrix(W_np.flatten(), W_pred_np.flatten())
    row, col = linear_sum_assignment(cm, maximize=True)

    # calculate cluster_accuracy
    cluster_acc = cm[row, col].sum()/cm.sum()

    return torch.tensor(col), cluster_acc

def get_permutation_and_graph(X_permute, Z_pred, Z_gt, dim_dict):
    X_hat = X_permute['X_hat'].detach().cpu().numpy()
    # mode_weights = X_permute['mode_weights'].detach().cpu().numpy()
    Z_gt = Z_gt.detach().cpu().numpy()
    mus = Z_pred['mus'].detach().cpu().numpy()
    nx = dim_dict['nx']
    ny = dim_dict['ny']
    num_nodes = dim_dict['num_nodes']
    num_variates = dim_dict['num_variates']
    lag = dim_dict['lag']

    # zt_true = np.einsum('bvtl,vnl->btn', X_lag, mode_weights)
    # mcc, sort_idx= compute_mcc(zt_recon.reshape((-1, num_nodes)).T, Z_gt.reshape((-1, num_nodes)).T, "Pearson")
    mcc, sort_idx = compute_mcc(mus[:,0].T, Z_gt[0,:-lag].T, "Pearson")

    sort_idx = sort_idx.numpy()   
    mus = mus[:,:, sort_idx]


    prediction = pcmci(mus[:,0,:], lag)
    # prediction = pcmci(zt_true[:,0,:], lag)
    prediction = np.array(prediction)
    G_predict = torch.from_numpy(prediction)

    
    return G_predict, mcc

def get_permutation(X_permute, dim_dict):
    X_lag = X_permute['X_lag'].detach().cpu().numpy()
    X_pred = X_permute['X_pred'].detach().cpu().numpy()
    Z_latent = X_permute['Z_latent'].detach().cpu().numpy()
    mode_weights = X_permute['mode_weights'].detach().cpu().numpy()
    nx = dim_dict['nx']
    ny = dim_dict['ny']
    num_nodes = dim_dict['num_nodes']
    num_variates = dim_dict['num_variates']
    lag = dim_dict['lag']

    X_0 = X_lag[0]
    mode_weights = mode_weights.reshape(num_variates, num_nodes, -1)
    X_latent = np.einsum("vtl,vnl->vtn", X_0, mode_weights)
    Z_0 = Z_latent[0]
    X_latent_0 = X_latent[0]
    C = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            C[i, j] = -pearsonr(X_latent_0[:, i], Z_0[:, j])[0]

    row_ind, col_ind = linear_sum_assignment(C)
    return torch.tensor(col_ind)


