"""
Acknowledgements: Lot of the code borrowed from https://github.com/microsoft/causica
and from https://github.com/xtibau/mapped_pcmci
"""

from pyro.distributions.transforms.spline import ConditionalSpline
from pyro.nn.dense_nn import DenseNN
from tqdm import tqdm
from torch import nn
from src.utils.data_generation.splines import unconstrained_RQS
import numpy as np
import torch

class PiecewiseRationalQuadraticTransform(nn.Module):
    """
    Layer that implements a spline-cdf (https://arxiv.org/abs/1906.04032) transformation.
     All dimensions of x are treated as independent, no coupling is used. This is needed
    to ensure invertibility in our additive noise SEM.

    Args:
        dim: dimensionality of input,
        num_bins: how many bins to use in spline,
        tail_bound: distance of edgemost bins relative to 0,
        init_scale: standard deviation of Gaussian from which spline parameters are initialised
    """

    def __init__(
        self,
        dim,
        num_bins=8,
        tail_bound=3.0,
        init_scale=1e-2,
    ):
        super().__init__()

        self.dim = dim
        self.num_bins = num_bins
        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_derivative = 1e-3
        self.tail_bound = tail_bound
        self.init_scale = init_scale

        self.params = nn.Parameter(
            self.init_scale * torch.randn(self.dim, self.num_bins * 3 - 1), requires_grad=True)

    def _piecewise_cdf(self, inputs, inverse=False):
        params_batch = self.params.unsqueeze(
            dim=(0)).expand(inputs.shape[0], -1, -1)

        unnormalized_widths = params_batch[..., : self.num_bins]
        unnormalized_heights = params_batch[...,
                                            self.num_bins: 2 * self.num_bins]
        unnormalized_derivatives = params_batch[..., 2 * self.num_bins:]

        return unconstrained_RQS(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            tail_bound=self.tail_bound,
        )

    def forward(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        return self._piecewise_cdf(inputs, inverse=False)

    def inverse(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        return self._piecewise_cdf(inputs, inverse=True)


def sample_none(input_dim):
    def func(X):
        return X
    return func


def sample_inverse_noise_spline(input_dim):
    flow = PiecewiseRationalQuadraticTransform(1, 16, 5, 1)
    W = np.ones((input_dim - 1, 1)) * (1 / (input_dim - 1))

    def func(X):
        z = X[..., :-1] @ W
        with torch.no_grad():
            y, _ = flow(torch.from_numpy(z))

            return -y.numpy() * X[..., -1:]

    return func


def sample_conditional_spline(input_dim):
    # input_dim is lagged_parent + 1
    noise_dim = 1
    count_bins = 8
    param_dim = [noise_dim * count_bins, noise_dim *
                 count_bins, noise_dim * (count_bins - 1)]
    hypernet = DenseNN(input_dim - 1, [20, 20], param_dim)
    transform = ConditionalSpline(
        hypernet, noise_dim, count_bins=count_bins, order="quadratic")

    def func(X):
        """
        X: lagged parents concat with noise. X[...,0:-1] lagged parents, X[...,-1] noise.
        """
        with torch.no_grad():
            X = torch.from_numpy(X).float()
            transform_cond = transform.condition(X[..., :-1])
            noise_trans = transform_cond(X[..., -1:])  # [batch, 1]
        return noise_trans.numpy()

    return func


def sample_spline(input_dim):
    flow = PiecewiseRationalQuadraticTransform(1, 16, 5, 1)
    W = np.ones((input_dim, 1)) * (1 / input_dim)

    def func(X):
        z = X @ W
        with torch.no_grad():
            y, _ = flow(torch.from_numpy(z))
            return y.numpy()

    return func


def sample_mlp(input_dim):
    mlp = DenseNN(input_dim, [64, 64], [1])
    coef = np.random.uniform(2,4)
    def func(X):
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            return coef*mlp(X).numpy()

    return func


def sample_mlp_noise(input_dim):
    mlp = DenseNN(input_dim - 1, [64, 64], [1])

    def func(X):
        X_pa = torch.from_numpy(X[..., :-1]).float()
        with torch.no_grad():
            return mlp(X_pa).numpy() * X[..., -1:]

    return func


def sample_spline_product(input_dim):
    flow = sample_spline(input_dim - 1)

    def func(X):
        z_p = flow(X[..., :-1])
        out = z_p * X[..., -1:]
        return out

    return func


def sample_linear(input_dim):

    # sample weights
    W = (2*np.random.binomial(n=1, p=0.5, size=(input_dim))-1) * \
        np.random.uniform(0.1,0.5, size=(input_dim))

    def func(X):
        return X@W[..., np.newaxis]

    return func

def sample_gaussian_linear(input_dim):

    # sample weights
    W = (2*np.random.binomial(n=1, p=0.5, size=(input_dim))-1) * \
        np.random.uniform(0.1, 0.5, size=(input_dim))
    
    # sample scaling factor for the Gaussian term
    A = np.random.uniform(1.0, 3.0)  # scaling factor for the Gaussian term
    B = np.random.uniform(1.0, 2.0)  # controls the spread of the Gaussian term

    def func(X):
        gaussian_term = A * np.exp(-B * np.sum(X**2, axis=-1) / 2)
        return X @ W[..., np.newaxis] * (1 + gaussian_term[..., np.newaxis])

    return func


def zero_func() -> np.ndarray:
    return np.zeros(1)
