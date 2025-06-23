import sys
import os
import time
import numpy as np
import scipy.stats


########## Function to import netcdf files

from netCDF4 import Dataset

def load_data(path_to_data, climate_variable):

        nc_fid = Dataset(path_to_data, 'r');
        climate_field = nc_fid.variables[climate_variable][:];
        print(climate_field.shape)
        return climate_field

def masked_array_to_numpy(data):
    return np.ma.filled(data.astype(np.float32), np.nan);

def get_nonmask_indices(data):
    return np.argwhere(~np.isnan(np.sum(data,axis=0)));

########## Compute lag correlations between two time series

# Note: time series are assumed to be normalized at zero mean and unit standard deviation
# x(t) = ( x(t) - Mean[ x(t)] ) / Std[ x(t) ]
def lagged_correlation(x,y,tau):

    assert len(x) == len(y);
    #length of time series
    T = len(x);

    ##assert that lag can nit be greater than T
    assert tau < T;

    if(tau == 0):
        return [tau,np.dot(x,y)/T];
    if(tau > 0):
        x = x[0:T-tau];
        y = y[tau:T]
        return [tau,np.dot(x,y)/T];

    if(tau < 0):
        tau = np.abs(tau)
        y = y[0:T-tau];
        x = x[tau:T]
        return [-tau,np.dot(x,y)/T];

# Get the correlations between timeseries ts1 and ts2 for a range of lags
def get_correlogram(ts1,ts2,tau_range):
    assert len(ts1) == len(ts2);
    correlogram = []
    for tau in range(-tau_range,tau_range+1):
        correlogram.append(lagged_correlation(ts1,ts2,tau))
    return np.array(correlogram)

########## Heuristic to estimate k

def estimate_k(flatten_data, rand_samples, q):

    # Input:
    # flatten_data: flatten dataset of dimension np.shape(flatten_data) = grid x time
    # rand_samples: number of sample to infer the k parameter
    # q: quantile to threshold sampled correlations

    # Output:
    # parameter k

    corrs = [];

    for i in range(rand_samples):

        ##sample two time series at random
        idx1 = np.random.randint(0, np.shape(flatten_data)[0],1);
        idx2 = np.random.randint(0, np.shape(flatten_data)[0],1);
        ts1 = flatten_data[idx1].squeeze();
        ts2 = flatten_data[idx2].squeeze();
        ##compute correlation at zero lag
        corr = lagged_correlation(ts1,ts2,0)[1]
        corrs.append(corr);

    corrs = np.array(corrs)
    k = np.quantile(corrs,q)

    return k

########## Heuristic to estimate eta
from sklearn.metrics import pairwise_distances

def estimate_eta_grid(flatten_data, coordinates, rand_samples, q):
    
    # Input:
    # flatten_data: flatten dataset of dimension np.shape(flatten_data) = grid x time
    # rand_samples: number of sample to infer the k parameter
    # q: quantile to threshold sampled correlations

    # Output:
    # parameter eta

    dist = [];

    for i in range(rand_samples):

        ##sample two time series at random
        idx1 = np.random.randint(0, np.shape(flatten_data)[0],1)[0];
        idx2 = np.random.randint(0, np.shape(flatten_data)[0],1)[0];
        # Compute distance
        dist.append(pairwise_distances(coordinates[idx1,None],coordinates[idx2,None])[0,0])

    dist = np.array(dist)
    eta = np.quantile(dist,q)

    return eta

def estimate_eta(flatten_data, coordinates, rand_samples, q):

    # Input:
    # flatten_data: flatten dataset of dimension np.shape(flatten_data) = grid x time
    # rand_samples: number of sample to infer the k parameter
    # q: quantile to threshold sampled correlations

    # Output:
    # parameter eta

    dist = [];

    for i in range(rand_samples):

        ##sample two time series at random
        idx1 = np.random.randint(0, np.shape(flatten_data)[0],1)[0];
        idx2 = np.random.randint(0, np.shape(flatten_data)[0],1)[0];
        # Compute distance
        dist.append(pairwise_distances(coordinates[idx1,None],coordinates[idx2,None],metric='haversine')[0,0])

    dist = np.array(dist)
    eta = np.quantile(dist,q)

    return eta
