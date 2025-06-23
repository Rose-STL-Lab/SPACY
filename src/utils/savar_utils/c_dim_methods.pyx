#!/usr/bin/env python

"""
This module contains all the dimensionality reduction methods used
"""

# Build in modules
from savar.functions import *

# C imports first
cimport numpy as np

# External modules
import numpy as np
from sklearn.linear_model import LinearRegression

# Type declarations
DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_int_t


# @@ VARIMAX PCA METHODS @@

cdef _pca_svd(np.ndarray[DTYPE_t, ndim=2] data,
              str truncate_by='max_comps',
              Py_ssize_t max_comps=60,
              float fraction_explained_variance=0.9,
              unsigned int verbosity=0):
    """

    :param data:
    :param truncate_by:
    :param max_comps:
    :param fraction_explained_variance:
    :param verbosity:
    :return:
    """

    """Assumes data of shape (obs, vars).

    https://stats.stackexchange.com/questions/134282/relationship-between-svd-
    and-pca-how-to-use-svd-to-perform-pca

    SVD factorizes the matrix A into two unitary matrices U and Vh, and a 1-D
    array s of singular values (real, non-negative) such that A == U*S*Vh,
    where S is a suitably shaped matrix of zeros with main diagonal s.

    K = min (obs, vars)

    U are of shape (vars, K)
    Vh are loadings of shape (K, obs)

    """

    # The actual function starts ####
    cdef unsigned int n_obs
    n_obs = data.shape[0]

    # Center data
    data -= data.mean(axis=0)

    # data_T = np.fastCopyAndTranspose(data)
    # print data.shape

    cdef np.ndarray[DTYPE_t, ndim=2] U
    cdef np.ndarray[DTYPE_t, ndim=1] s
    cdef np.ndarray[DTYPE_t, ndim=2] Vt

    U, s, Vt = np.linalg.svd(data,
                             full_matrices=False)
    # False, True, True)

    # flip signs so that max(abs()) of each col is positive
    U, Vt = _svd_flip(U, Vt, u_based_decision=False)

    V = Vt.T
    cdef np.ndarray[DTYPE_t, ndim=2] S
    S = np.diag(s)

    # eigenvalues of covariance matrix
    cdef np.ndarray[DTYPE_t, ndim=1] eig
    eig = (s ** 2) / (n_obs - 1.)

    # Sort
    cdef np.ndarray[DTYPE_int_t, ndim=1] idx
    idx = eig.argsort()[::-1]
    eig, U = eig[idx], U[:, idx]

    cdef float explained
    if truncate_by == 'max_comps':

        U = U[:, :max_comps]
        V = V[:, :max_comps]
        S = S[0:max_comps, 0:max_comps]
        explained = np.sum(eig[:max_comps]) / np.sum(eig)

    elif truncate_by == 'fraction_explained_variance':
        # print np.cumsum(s2)[:80] / np.sum(s2)
        max_comps = np.argmax(np.cumsum(eig) / np.sum(eig) > fraction_explained_variance) + 1
        explained = np.sum(eig[:max_comps]) / np.sum(eig)

        U = U[:, :max_comps]
        V = V[:, :max_comps]
        S = S[0:max_comps, 0:max_comps]

    else:
        max_comps = U.shape[1]
        explained = np.sum(eig[:max_comps]) / np.sum(eig)

    # Time series
    cdef np.ndarray[DTYPE_t, ndim=2] ts
    ts = U.dot(S)

    return V, U, S, ts, eig, explained, max_comps


cdef _svd_flip(u, v=None, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u, v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if v is None:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_rows, range(u.shape[1])])
        u *= signs
        return u

    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]

    return u, v


def varimax(np.ndarray[DTYPE_t, ndim=2] Phi,
            float gamma=1.0,
            unsigned int q=500,
            float rtol=np.finfo(np.float32).eps ** 0.5,
            unsigned int verbosity=0):
    """

    :param Phi: The V of the SVD
    :param gamma: if = 1, equals to Varimax. if = 0 quartimax. if = k/2 equamax. if = p*(k-1)/(p+k-2) parsimax
    :param q: number of iterations, breaks if objective archived before
    :param rtol: parameter of the machine
    :param verbosity: verbosity
    :return: Rotated Phi and Rotation matrix
    """

    # Some definitions
    cdef unsigned int p
    cdef unsigned int k
    cdef float d = 0.
    cdef float d_old
    cdef np.ndarray[DTYPE_t, ndim=2] R
    cdef np.ndarray[DTYPE_t, ndim=2] Lambda

    p = Phi.shape[0]
    k = Phi.shape[1]
    R = np.eye(k)

    # print Phi
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=2] u
    cdef np.ndarray[DTYPE_t, ndim=1] s
    cdef np.ndarray[DTYPE_t, ndim=2] vh

    for i in range(q):
        if verbosity > 1:
            if i % 10 == 0.:
                print("\t\tVarimax iteration %d" % i)
        d_old = d
        Lambda = np.dot(Phi, R)

        u, s, vh = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda) ** 3
                                        - (gamma / float(p)) * np.dot(Lambda,
                                                                      np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and abs(d - d_old) / d < rtol:
            break

    return np.dot(Phi, R), R


def get_varimax_loadings_standard(data,
                                  truncate_by='max_comps',
                                  max_comps=60,
                                  fraction_explained_variance=0.9,
                                  verbosity=True,
                                  ):
    if verbosity:
        print("Get Varimax components")
        print("\t Get SVD")

    data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:])))  # flattening field of daily data

    # Get truncated SVD
    V, U, S, ts_svd, eig, explained, max_comps = _pca_svd(data=data,
                                                               truncate_by=truncate_by, max_comps=max_comps,
                                                               fraction_explained_variance=fraction_explained_variance,
                                                               verbosity=verbosity)
    # if verbosity > 0:
    #     print("Explained variance at max_comps = %d: %.5f" % (max_comps, explained))

    if verbosity:
        if truncate_by == 'max_comps':

            print("\t User-selected number of components: %d\n"
                  "\t Explaining %.2f of variance" % (max_comps, explained))

        elif truncate_by == 'fraction_explained_variance':

            print("\t User-selected explained variance: %.2f of total variance\n"
                  "\t Resulting in %d components" % (explained, max_comps))

    if verbosity:
        print("\t Varimax rotation")
    # Rotate
    Vr, Rot = varimax(V, verbosity=verbosity)
    # Vr = V
    # Rot = np.diag(np.ones(V.shape[1]))
    # print Vr.shape
    Vr = _svd_flip(Vr)

    if verbosity:
        print("\t Further metrics")
    # Get explained variance of rotated components
    s2 = np.diag(S) ** 2 / (data.shape[0] - 1.)

    # matrix with diagonal containing variances of rotated components
    S2r = np.dot(np.dot(np.transpose(Rot), np.matrix(np.diag(s2))), Rot)
    expvar = np.diag(S2r)

    sorted_expvar = np.sort(expvar)[::-1]
    # s_orig = ((Vt.shape[1] - 1) * s2) ** 0.5

    # reorder all elements according to explained variance (descending)
    nord = np.argsort(expvar)[::-1]
    Vr = Vr[:, nord]

    # Get time series of UNMASKED data
    comps_ts = data.dot(Vr)

    total_var = np.sum(np.var(data, axis=0))

    return {'weights': np.copy(Vr),
            'explained_var': sorted_expvar,
            'unrotated_weights': V,
            'explained': explained,
            'pca_eigs': eig,
            'comps_ts': comps_ts,
            'truncate_by': truncate_by,
            'max_comps': max_comps,
            'fraction_explained_variance': fraction_explained_variance,
            'total_var': total_var,
            }

def remove_past_influence(np.ndarray[DTYPE_t, ndim = 2] data, unsigned int sample = 0, bint verbose=True):

    """
    :param data shpe (time, lat x lon)
    """
    # Shape
    cdef unsigned int t = data.shape[0]
    cdef unsigned int size = data.shape[1]

    # Elements of the regression
    cdef np.ndarray[DTYPE_t, ndim=1] y
    cdef np.ndarray[DTYPE_t, ndim=1] y_fast
    cdef np.ndarray[DTYPE_t, ndim=2] X
    cdef np.ndarray[DTYPE_t, ndim=2] X_fast
    cdef np.ndarray[DTYPE_t, ndim=2] results = np.zeros((t-1, size))

    # First we create the first element
    cdef Py_ssize_t i
    for i in range(size):
        if verbose:
            print("Regression in {}%".format(100*i/size))

        # Condition on the previous time-steps of other variables
        y = data[1:, i]
        X = np.delete(data[:-1, :], i, axis=1)

        # In case of using a sample to determine the regression
        if sample != 0:
            y_fast = data[1:sample, i]
            X_fast = np.delete(data[:sample-1, :], i, axis=1)
            results[:, i] = y - LinearRegression().fit(X_fast, y_fast).predict(X)
        else:
            results[:, i] = y- LinearRegression().fit(X, y).predict(X)

    return results
