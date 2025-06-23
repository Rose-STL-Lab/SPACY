# External imports
import numpy as np
from copy import deepcopy
from math import sin, pi
import itertools as it  # TODO REmove

# Interal imports
from src.utils.savar_utils.functions import check_stability
# from src.utils.savar_utils.eval_tools import Evaluation, DmMethod


### TODO: To be moves ###
def dict_to_matrix(links_coeffs, default=0):
    """
    Maps to the coefficient matrix. Without time
    :param links_coeffs:
    :param default:
    :return: a matrix coefficient of [j, i, \tau] where a link is i -> j at \tau
    """
    tau_max = max(abs(lag)
                  for (_, lag), _ in it.chain.from_iterable(links_coeffs.values()))

    n_vars = len(links_coeffs)

    graph = np.ones((n_vars, n_vars, tau_max+1), dtype=float)
    graph *= default

    for j, values in links_coeffs.items():
        for (i, tau), coeff in values:
            graph[j, i, abs(tau)] = coeff
    return graph

def matrix_to_dict(matrix, link_strength=0.3):
    """
    Converts a coefficient matrix back into a dictionary representation.
    :param matrix: A 3D numpy array representing the coefficient matrix.
    :param default: The default value to ignore (usually 0).
    :return: A dictionary representation of links and coefficients in the desired format.
    """
    links_coeffs = {}
    tau_max, _, n_vars  = matrix.shape
    for j in range(n_vars):
        link_list = []
        for tau in range(tau_max):
            for i in range(n_vars):
                coeff = matrix[tau,j,i]
                if coeff != 0:
                    link_list.append(([i, -tau], link_strength*coeff))
        links_coeffs[j] = link_list

    return links_coeffs

def dict_to_matrixV0(links_coeffs, default=0):
    """
    Maps to the coefficient matrix. Without time
    :param links_coeffs:
    :param default:
    :return: a matrix coefficient of [j, i, \tau-1] where a link is i -> j at \tau
    """
    tau_max = max(abs(lag)
                  for (_, lag), _ in it.chain.from_iterable(links_coeffs.values()))

    n_vars = len(links_coeffs)

    graph = np.ones((n_vars, n_vars, tau_max), dtype=float)
    graph *= default

    for j, values in links_coeffs.items():
        for (i, tau), coeff in values:
            graph[j, i, abs(tau) - 1] = coeff

    return graph


###


class SAVAR:
    """
    Main class containing SAVAR model
    """

    __slots__ = ["links_coeffs", "n_vars", "time_length", "transient", "spatial_resolution", "tau_max",
                 "mode_weights", "noise_weights",
                 "noise_cov", "noise_strength", "noise_variance", "latent_noise_cov", "fast_noise_cov",
                 "forcing_dict", "season_dict",
                 "data_field", "noise_data_field", "seasonal_data_field", "forcing_data_field",
                 "linearity", "verbose", "model_seed"]

    def __init__(self, links_coeffs: dict, time_length: int, mode_weights: np.ndarray, transient: int = 200,
                 noise_weights: np.ndarray = None,
                 noise_strength: float = 1, noise_variance: float = 1, noise_cov: np.ndarray = None,
                 latent_noise_cov: np.ndarray = None, fast_cov: np.ndarray = None,
                 forcing_dict: dict = None, season_dict: dict = None,
                 data_field: np.ndarray = None, noise_data_field: np.ndarray = None,
                 seasonal_data_field: np.ndarray = None, forcing_data_field: np.ndarray = None,
                 linearity: str = "linear", verbose: bool = False, model_seed: int = None,
                 ):

        self.links_coeffs = links_coeffs
        self.time_length = time_length
        self.transient = transient
        self.noise_strength = noise_strength
        self.noise_variance = noise_variance  #TODO: NOT USED.
        self.noise_cov = noise_cov

        self.latent_noise_cov = latent_noise_cov  # D_x
        self.fast_noise_cov = fast_cov  # D_y

        self.mode_weights = mode_weights
        self.noise_weights = noise_weights

        self.forcing_dict = forcing_dict
        self.season_dict = season_dict

        self.data_field = data_field

        self.linearity = linearity
        self.verbose = verbose
        self.model_seed = model_seed

        # Computed attributes
        self.n_vars = len(links_coeffs)
        self.tau_max = max(abs(lag)
                           for (_, lag), _ in it.chain.from_iterable(self.links_coeffs.values()))
        self.spatial_resolution = deepcopy(self.mode_weights.reshape(self.n_vars, -1).shape[1])

        if self.noise_weights is None:
            self.noise_weights = deepcopy(self.mode_weights)
        if self.latent_noise_cov is None:
            self.latent_noise_cov = np.eye(self.n_vars)
        if self.fast_noise_cov is None:
            # TODO: Should be np.eye ???
            self.fast_noise_cov = np.zeros((self.spatial_resolution, self.spatial_resolution))

        # Empty attributes
        self.noise_data_field = noise_data_field
        self.seasonal_data_field = seasonal_data_field
        self.forcing_data_field = forcing_data_field

        if np.random is not None:
            np.random.seed(model_seed)

    def generate_data(self) -> None:
        """
        Generates the data of savar
        :return:
        """
        # Prepare the datafield
        if self.data_field is None:
            if self.verbose:
                print("Creating empty data field")
            # Compute the field
            self.data_field = np.zeros((self.spatial_resolution, self.time_length + self.transient))

        # Add noise first
        if self.noise_data_field is None:
            if self.verbose:
                print("Creating noise data field")
            self._add_noise_field()
        else:
            self.data_field += self.noise_data_field

        # Add seasonality
        if self.season_dict is not None:
            if self.verbose:
                print("Adding seasonality forcing")
            self._add_seasonality_forcing()

        # Add external forcing
        if self.forcing_dict is not None:
            if self.verbose:
                print("Adding external forcing")
            self._add_external_forcing()

            # Compute the data
        if self.linearity == "linear":
            if self.verbose:
                print("Creating linear data")
            self._create_linear()
        else:
            raise NotImplemented("Now, only linear methods are implemented")

    def generate_cov_noise_matrix(self) -> np.ndarray:
        """
        W \in NxL
        data_field L times T

        :return:
        """

        W = deepcopy(self.noise_weights).reshape(self.n_vars, -1)
        W_plus = np.linalg.pinv(W)
        cov = self.noise_strength * W_plus @ self.latent_noise_cov @ W_plus.transpose() + self.fast_noise_cov

        return cov

    def _add_noise_field(self):
        if self.noise_cov is None:
            self.noise_cov = self.generate_cov_noise_matrix()

        # Generate noise from cov
        self.noise_data_field = np.random.multivariate_normal(mean=np.zeros(self.spatial_resolution), cov=self.noise_cov,
                                                              size=self.time_length + self.transient).transpose()

        self.data_field += self.noise_data_field

    def _add_seasonality_forcing(self):

        # A*sin((2pi/lambda)*x) A = amplitude, lambda = period
        amplitude = self.season_dict["amplitude"]
        period = self.season_dict["period"]
        season_weight = self.season_dict.get("season_weight", None)

        seasonal_trend = np.asarray([amplitude * sin((2 * pi / period) * x)
                                     for x in range(self.time_length +
                                                    self.transient)])

        seasonal_data_field = np.ones_like(self.data_field)
        seasonal_data_field *= seasonal_trend.reshape(1, -1)

        # Apply seasonal weights
        if season_weight is not None:
            season_weight = season_weight.sum(axis=0).reshape(self.spatial_resolution)  # vector dim L
            seasonal_data_field *= season_weight[:, None]  # L times T

        self.seasonal_data_field = seasonal_data_field

        # Add it to the data field.
        self.data_field += seasonal_data_field

    def _add_external_forcing(self):

        if self.forcing_dict is None:
            raise TypeError("Forcing dict is empty")

        w_f = deepcopy(self.forcing_dict.get("w_f"))
        f_1 = deepcopy(self.forcing_dict.get("f_1"))
        f_2 = deepcopy(self.forcing_dict.get("f_2"))
        f_time_1 = deepcopy(self.forcing_dict.get("f_time_1"))
        f_time_2 = deepcopy(self.forcing_dict.get("f_time_2"))

        if w_f is None:
            w_f = deepcopy(self.mode_weights)
            w_f = w_f.astype(bool).astype(int)  # Converts non-zero elements of the weight into 1.

        w_f_sum = w_f.sum(axis=0)
        f_time_1 += self.transient
        f_time_2 += self.transient

        # Check
        time_length = self.time_length + self.transient
        trend = np.concatenate((np.repeat([f_1], f_time_1), np.linspace(f_1, f_2, f_time_2 - f_time_1),
                                np.repeat([f_2], time_length - f_time_2))).reshape((1, time_length))

        forcing_field = (w_f_sum.reshape(1, -1) * trend.transpose()).transpose()
        self.forcing_data_field = forcing_field

        # Add it to the data field.
        self.data_field += forcing_field

    def _create_linear(self):

        """
            weights N \times L
            data_field L \times T
        """
        weights = deepcopy(self.mode_weights.reshape(self.n_vars, -1))
        weights_inv = np.linalg.pinv(weights)
        time_len = deepcopy(self.time_length)
        time_len += self.transient
        tau_max = self.tau_max

        phi = dict_to_matrix(self.links_coeffs)
        data_field = deepcopy(self.data_field)

        for t in range(tau_max, time_len):
            for i in range(tau_max):
                data_field[..., t:t + 1] += weights_inv @ phi[..., i] @ weights @ data_field[..., t - 1 - i:t - i]

        self.data_field = data_field[..., self.transient:]


