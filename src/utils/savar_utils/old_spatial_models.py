# This module contains spatio-temporal models

import numpy as np
import matplotlib.pyplot as plt
import copy
from savar.functions import make_dir, create_random_mode, check_stability, is_pos_def
from savar.c_functions import create_graph, compute_linear_savar, generate_cov_matrix, compute_nonlinear_linear_savar, \
    find_max_lag
from math import pi, sin
import matplotlib.pyplot as plt


class savarModel:
    """
    This class holds the savar model with all the information
    """

    # Use __slot__ to save memory
    __slots__ = ['links_coeffs', 'noise_weights', 'random_noise', 'max_loop_noise', 'modes_weights', 'variance_noise',
                 'covariance_noise_method', 'ny', 'nx', 'T', 'spatial_covariance', 'transient', 'n_variables',
                 'verbose', 'season', 'season_weight', 'forcing_field', 'lineartiy', 'nonl_coeff', 'w_t', 'data_field',
                 'network_data', 'forcing', 'cov', 'noise_field', 'w_f']

    def __init__(self, links_coeffs: dict, modes_weights: np.ndarray, ny: int, nx: int, T: int,
                 spatial_covariance: float, noise_weights: np.ndarray = None,
                 variance_noise: float = 1, covariance_noise_method: str = "weights_transposed",
                 max_loop_noise: int = 10, random_noise: tuple = (0, 1), transient: int = 200, forcing: tuple = None,
                 n_variables: int = 3, season: tuple = None, season_weight: np.ndarray = None,
                 forcing_field: np.ndarray = None, linearity: str = "linear", nonl_coeff: float = 0.5,
                 w_t: bool = False, verbose: bool = True) -> None:

        """
        :type model_des_uniform:
        :param links_coeffs:
        :param weights:
        :param variance_noise dets de value of the diagonal og the correlation noise \Sigma
        :param Defines how to map from noise weights to \sigma.
        :param ny (L): ny, number of longitudinal grid points
        :param nx (K): nx: number of latitudinal grid points
        :param T: number of time-steps of the model
        :param spatial_covariance: used to create the noise covariance. Decrease it if is not pos. semidef
        :param forcing: (forcing weight matrix: np.ndarray(n_var, ny, nx),
                          forcing value at short term: float, forcing at long term: float,
                          time-step till F = f_1: int, time-step which f = f_2: int)
        :param n_variables: how many climate variables are there, asssuming that they are longitudinally concatenated
        :param season (amplitude, period): define a seasonality on data
        :param season_weight np.ndarray (ny, nx) if None and season not None, is 1 everywhere
        :param random_noise (mean, var) or None
        :param forcing_field: some external forcing that is added tot the noise field. shape: (T, nx*ny)
        :param linearity: "linear" or "non-linear"
        :param w_t: A matrix with W as a function of time.
        :param verbose:
        """
        # TODO: Make forcing dictionary.
        # Define attributes
        self.links_coeffs = links_coeffs
        self.noise_weights = noise_weights
        self.random_noise = random_noise
        self.max_loop_noise = max_loop_noise
        self.modes_weights = modes_weights
        self.variance_noise = variance_noise
        self.covariance_noise_method = covariance_noise_method
        self.ny = ny
        self.nx = nx
        self.T = T
        self.spatial_covariance = spatial_covariance
        self.transient = transient
        self.n_variables = n_variables
        self.verbose = verbose
        self.season = season
        self.season_weight = season_weight
        self.forcing_field = forcing_field
        self.lineartiy = linearity
        self.nonl_coeff = nonl_coeff
        self.w_t = w_t
        if self.w_t:
            self.transient = 0

        # Data
        self.data_field = None
        self.network_data = None
        self.noise_field = None

        # Check if there is forcing and assign it
        if forcing is not None:
            self.forcing = {'w_f': forcing[0], 'f_1': forcing[1],
                            'f_2': forcing[2], 'f_time_1': forcing[3],
                            'f_time_2': forcing[4]
                            }
        else:
            self.forcing = None

        # Set noise weights '= mode weights
        if self.noise_weights is None:
            self.noise_weights = copy.deepcopy(self.modes_weights)

    def create_linear_savar_data(self) -> None:

        """
        Creates the linear data of  savar model
        """

        # To consider or not forcing
        forcing = True if self.forcing is not None else False

        if forcing:
            w_f, f_1, f_2, f_time_1, f_time_2 = self.forcing['w_f'], self.forcing['f_1'], \
                                                self.forcing['f_2'], self.forcing['f_time_1'], self.forcing['f_time_2']
            # The default value of w_f is 1 everywhere
            w_f = w_f if w_f is not None else copy.deepcopy(self.modes_weights)
            w_f = w_f.sum(axis=0)

            self.w_f = w_f

            assert self.T > f_time_2, "F_time_2 should be smaller than T"

        # Obtain some useful parameters
        graph, max_lag = create_graph(self.links_coeffs, return_lag=True)
        n_var = graph.shape[0]

        if self.verbose:
            print("Start...")

        # For facility in coding remove the self
        nx, ny, T, transient = self.nx, self.ny, self.T, self.transient

        # Compute data_field \in R^(T+transient, nx, ny)
        if self.noise_weights is not None:
            # Covariance matrix
            cov = generate_cov_matrix(noise_weights=self.noise_weights, spatial_covariance=self.spatial_covariance,
                                      variance=self.variance_noise, method=self.covariance_noise_method)

            self.cov = cov

            data_field = np.random.multivariate_normal(mean=np.zeros(ny * nx), cov=cov, size=T + transient,
                                                       check_valid="ignore")

            self.noise_field = copy.deepcopy(data_field)[transient:, ...]
        else:
            raise NotImplemented("noise_weights cannot be none")

        if forcing:
            if self.verbose:
                print("Adding external forcing...")
            # Add the trend. concat 1st period, rising, 2nd trend
            trend = np.concatenate((np.repeat([f_1], f_time_1 + transient), np.linspace(f_1, f_2, f_time_2 - f_time_1),
                                    np.repeat([f_2], T - f_time_2))).reshape((1, T + transient))

            print("Printing shape trend. {}".format(trend.shape))

            # We modify the data_field to add all the forcing (season an external)
            external_forcing_field = (w_f.reshape((ny * nx, 1)) * trend)  # \in R^(nx*ny, T+transient)

        if self.season is not None:
            if self.verbose:
                print("Adding seasonality...")
            # A*sin((2pi/lambda)*x) A = amplitude, lambda = period
            amplitude, period = self.season
            seasonal_trend = np.asarray([amplitude * sin((2 * pi / period) * x)
                                         for x in range(T + transient)]).reshape((1, 1, T + transient))
            # use seasonal weight if it is none, then equal for each grid point
            season_weight = np.ones((nx * ny)) if self.season_weight is \
                                                  None else self.season_weight.reshape(nx * ny)
            seasonal_forcing_field = (season_weight.reshape(ny * nx, 1) * seasonal_trend).reshape(nx * ny,
                                                                                                  T + transient)

            # TODO: comprobar dimensions abans de sumar!!!!

            if self.verbose:
                print("data_field shape is:", data_field.shape)
                print("seasonal_forcing_field shape is:", seasonal_forcing_field.shape)

            # Add the forcings to field
            data_field += seasonal_forcing_field.transpose()

        if forcing:
            print(data_field.shape)
            print(external_forcing_field.transpose().shape)
            data_field += external_forcing_field.transpose()

        if self.verbose:
            print("Compute values in time...")

        # Add other external forcing from self.forcing_field
        if self.forcing_field is not None:
            data_field[transient:, ...] += self.forcing_field

        if not self.w_t:  # Not dinamical W
            data_field, network = compute_linear_savar(data_field,
                                                       self.modes_weights.reshape(n_var, nx * ny).transpose(),
                                                       graph, w_time_dependant=False)
        else:
            data_field, network = compute_linear_savar(data_field_noise=data_field,
                                                       weights=np.transpose(
                                                           self.modes_weights.reshape((T, n_var, nx * ny)),
                                                           axes=(0, 2, 1)),
                                                       graph=graph, w_time_dependant=True)

        if self.verbose:
            print("Done...")

        # Return cutting off transient
        self.data_field = data_field[transient:, :]
        self.network_data = network[transient:]

    def create_nonlinear_savar_data(self) -> None:

        """
        Creates the non-linear data of  savar model
        The link_coeff its includes the type of non linearty
        """

        # To consider or not forcing
        forcing = True if self.forcing is not None else False

        if forcing:
            w_f, f_1, f_2, f_time_1, f_time_2 = self.forcing['w_f'], self.forcing['f_1'], \
                                                self.forcing['f_2'], self.forcing['f_time_1'], self.forcing['f_time_2']

        # Some checks
        if forcing:
            assert self.T > f_time_2, "F_time_2 should be bigger than T"

        # Obtain some useful parameters

        # max_lag = find_max_lag(self.links_coeffs)
        n_var = len(self.links_coeffs)

        if self.verbose:
            print("Start...")

        # For facility in coding remove the self
        nx, ny, T, transient = self.nx, self.ny, self.T, self.transient

        # Compute data_field \in R^(T+transient, nx, ny)
        if self.noise_weights is not None:
            # Covariance matrix
            cov = generate_cov_matrix(noise_weights=self.noise_weights, spatial_covariance=self.spatial_covariance,
                                      variance=self.variance_noise, method=self.covariance_noise_method)
            self.cov = cov

            data_field = np.random.multivariate_normal(mean=np.zeros(ny * nx), cov=cov, size=T + transient,
                                                       check_valid="ignore")
        else:
            noise_mean, noise_variance = self.random_noise
            data_field = (np.random.rand(ny * nx * (T + transient)).reshape(
                (T + transient, nx * ny)) * noise_variance) + noise_mean

        if forcing:
            if self.verbose:
                print("Adding external forcing...")

            external_forcing_field = self._add_external_forcing()

        if self.season is not None:
            if self.verbose:
                print("Adding seasonality...")
            # A*sin((2pi/lambda)*x) A = amplitude, lambda = period
            amplitude, period = self.season
            seasonal_trend = np.asarray([amplitude * sin((2 * pi / period) * x)
                                         for x in range(T + transient)]).reshape((1, 1, T + transient))
            # use seasonal weight if it is none, then equal for each grid point
            season_weight = np.ones((nx * ny)) if self.season_weight is \
                                                  None else self.season_weight.reshape(nx * ny)
            seasonal_forcing_field = (season_weight.reshape(ny * nx, 1) * seasonal_trend).reshape(nx * ny,
                                                                                                  T + transient)

            # TODO: comprobar dimensions abans de sumar!!!!

            if self.verbose:
                print("data_field shape is:", data_field.shape)
                print("seasonal_foorcing_field shape is:", seasonal_forcing_field.shape)

            # Add the forcings to field
            data_field += seasonal_forcing_field.transpose()

        if forcing:
            data_field += external_forcing_field.transpose()

        if self.verbose:
            print("Compute values in time...")

        # Add other external forcing from self.forcing_field
        if self.forcing_field is not None:
            data_field[transient:, ...] += self.forcing_field

        data_field, network = compute_nonlinear_linear_savar(data_field,
                                                             self.modes_weights.reshape(n_var, nx * ny).transpose(),
                                                             self.links_coeffs,
                                                             self.nonl_coeff)

        if self.verbose:
            print("Done...")

        # Return cutting off transient
        self.data_field = data_field[transient:, :]
        self.network_data = network[transient:]

    def create_savar_data(self):
        if self.lineartiy == "linear":
            self.create_linear_savar_data()
        elif self.lineartiy == "nonlinear":
            self.create_nonlinear_savar_data()
        else:
            raise KeyError("linearty must be linear or nonlinear. currently", self.lineartiy)

    def _add_external_forcing(self):

        """
        Returns a forcing field of shape T \time L according to dictionary self.forcing.
        :return:
        """

        w_f, f_1, f_2, f_time_1, f_time_2 = self.forcing['w_f'], self.forcing['f_1'], \
                                            self.forcing['f_2'], self.forcing['f_time_1'], self.forcing['f_time_2']

        w_f_sum = w_f.sum(axis=0)

        trend = np.concatenate((np.repeat([f_1], f_time_1 + self.transient), np.linspace(f_1, f_2, f_time_2 - f_time_1),
                                np.repeat([f_2], self.T - f_time_2))).reshape((1, self.T + self.transient))

        return w_f_sum.reshape(1, -1) * trend.transpose()

    def plot(self):
        plt.plot(self.network_data)
        plt.show()
