# Functions to generate random models
import numpy as np
import itertools as it
import random
from copy import deepcopy
from typing import Tuple
import warnings

# Internal
from src.utils.savar_utils.functions import create_random_mode, create_non_stationarity
from src.utils.savar_utils.savar import SAVAR


class SavarGenerator:

    def __init__(self,
                 links_coeffs: dict = None, n_variables: int = 3, time_length: int = 500, transient: int = 200,
                 # Noise
                 noise_strength: float = 1., noise_variance: float = None, noise_weights: np.ndarray = None,
                 resolution: tuple = (10, 10), noise_cov: np.ndarray = None,
                 latent_noise_cov: np.ndarray = None, fast_cov: np.ndarray = None,
                 # Fields
                 data_field: np.ndarray = None, noise_data_field: np.ndarray = None,
                 seasonal_data_field: np.ndarray = None, forcing_data_field: np.ndarray = None,
                 # Weights
                 mode_weights: np.ndarray = None,  gaussian_shape: bool = True,
                 random_mode=True, dipole: int = None, mask_variables: int = None, norm_weight: bool = True,
                 # links
                 n_cross_links=3, auto_coeffs_mean: float = 0.3, auto_coffs_std: float = 0.2,
                 auto_links: bool = True,
                 auto_strength_threshold: float = 0.2, auto_random_sign: float = 0.,
                 cross_mean: float = 0.3, cross_std: float = 0.2, cross_threshold: float = 0.2,
                 cross_random_sign: float = 0.2, tau_max: int = 3, tau_min: int = 1,
                 n_trial: int = 1000, model_seed: int = None,
                 # external forcings
                 forcing_dict: dict = None, season_dict: dict = None,
                 # Ornstein
                 ornstein_sigma=None, n_var_ornstein=None,
                 # Lineartiy
                 linearity="linear",
                 # Verbose
                 verbose=False
                 ):
        """

        :param links_coeffs:
        :param n_variables:
        :param time_length:
        :param transient:
        :param noise_strength:
        :param noise_variance:
        :param noise_weights:
        :param resolution:
        :param gaussian_shape:
        :param mode_weights:
        :param random_mode:
        :param dipole:
        :param mask_variables:
        :param norm_weight:
        :param n_cross_links:
        :param auto_coeffs_mean:
        :param auto_coffs_std:
        :param auto_links:
        :param auto_strength_threshold:
        :param auto_random_sign:
        :param cross_mean:
        :param cross_std:
        :param cross_threshold:
        :param cross_random_sign:
        :param tau_max:
        :param n_trial:
        :param model_seed:
        :param forcing_dict:
        :param season_dict:
        :param ornstein_sigma:
        :param n_var_ornstein:
        :param linearity:
        :param verbose:
        """
        # Basic
        self.links_coeffs = links_coeffs
        self.n_variables = n_variables
        self.time_length = time_length
        self.transient = transient

        # Noise
        self.noise_strength = noise_strength
        self.noise_variance = noise_variance
        self.noise_weights = noise_weights
        self.noise_cov = noise_cov
        self.latent_noise_cov = latent_noise_cov
        self.fast_cov = fast_cov
        self.resolution = resolution
        self.gaussian_shape = gaussian_shape

        # Fields
        self.data_field = data_field
        self.noise_data_field = noise_data_field
        self.seasonal_data_field = seasonal_data_field
        self.forcing_data_field = forcing_data_field

        # Weights
        self.mode_weights = mode_weights
        self.random_mode = random_mode
        self.dipole = dipole
        self.mask_variables = mask_variables
        self.norm_weight = norm_weight

        # links
        self.n_cross_links = n_cross_links
        self.auto_coeffs_mean = auto_coeffs_mean
        self.auto_coffs_std = auto_coffs_std
        self.auto_links = auto_links
        self.auto_strength_threshold = auto_strength_threshold
        self.auto_random_sign = auto_random_sign
        self.cross_mean = cross_mean
        self.cross_std = cross_std
        self.cross_threshold = cross_threshold
        self.cross_random_sign = cross_random_sign
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.n_trial = n_trial
        self.model_seed = model_seed

        # external forcings
        self.forcing_dict = forcing_dict
        self.season_dict = season_dict

        # Ornstein
        self.ornstein_sigma = ornstein_sigma
        self.n_var_ornstein = n_var_ornstein

        # Lineartiy
        self.linearity = linearity

        # Verbose
        self.verbose = verbose

        # Some checks
        if self.tau_min < 1 & self.tau_min > self.tau_max:
            raise KeyError("Tau min must be at least one and smaller or equal than tau_max")

        if noise_weights is None and mode_weights is not None:
            self.noise_weights = self.mode_weights

        if mode_weights is None and mode_weights is not None:
            self.mode_weights = noise_weights

        if mode_weights is not None:
            if self.mode_weights.reshape(n_variables, -1).shape[0] != resolution[0]*resolution[1]:
                if self.verbose:
                    print("Warning: changing resolution to the shape of the modes")
                self.resolution = (self.mode_weights.shape[1], self.mode_weights.shape[2])
                
        if self.links_coeffs is not None:
            n_variables = len(self.links_coeffs)
            if self.n_variables != n_variables:
                warnings.warn("Number of variables in links_coeffs is different from n_vars."
                              " Setting it to links_coeffs")
                self.n_variables = n_variables

        if self.links_coeffs is not None:
            n_variables = len(self.links_coeffs)
            if self.n_variables != n_variables:
                warnings.warn("Number of variables in links_coeffs is different from n_vars."
                              " Setting it to links_coeffs")
                self.n_variables = n_variables

        if self.verbose:
            print("Class model generator created")

    def generate_links_coeff(self):
        """
        Generates the random links coeffs according to the input
        :return:
        """
        np.random.seed(self.model_seed)
        random.seed(self.model_seed)
        
        # Create empty link list
        links = {N: [] for N in range(self.n_variables)}

        # Autocorrelation
        if self.auto_links:
            for i in range(self.n_variables):
                links[i].append(((int(i), -1), self._get_random_link_strength(mean=self.auto_coeffs_mean,
                                                                              std=self.auto_coffs_std,
                                                                              strength_threshold=self.auto_strength_threshold,
                                                                              random_sign=self.auto_random_sign)))

        # Cross links
        # Generate couplings
        all_possible = np.array(list(it.permutations(range(self.n_variables), 2)))

        all_possible_tau = [(i, j, tau) for i, j in all_possible for tau in range(self.tau_min, self.tau_max + 1)]

        # Choose n_links cross links at different taus
        random.shuffle(all_possible_tau)
        chosen_links = all_possible_tau[:self.n_cross_links]

        for (i, j, tau) in chosen_links:
            coeff = self._get_random_link_strength(mean=self.cross_mean, std=self.cross_std,
                                                   strength_threshold=self.cross_threshold,
                                                   random_sign=self.cross_random_sign)
            links[j].append(((i, -tau), coeff))

        # Stationarity check assuming model with linear dependencies at least for large x
        if self.check_stationarity(links)[0]:
            if self.linearity == "linear":
                return links
            else:
                raise NotImplementedError("Non linear models are not implemented")
        else:
            return None

    def generate_weights(self) -> Tuple[int, np.ndarray]:

        # First estimate the size of of each mode and its location
        ny, nx = self.resolution
        size, positions = self.find_mode_positions(res=self.resolution, n_var=self.n_variables)
        weights = np.zeros((self.n_variables, ny, nx))

        if self.dipole is not None:  # If we include dipoles
            dipol_list = random.sample(list(range(self.n_variables)), self.dipole)
            for i in range(self.n_variables):
                y_1, y_2, x_1, x_2 = positions[i]
                if i in dipol_list:  # If this variable is a dipole
                    weights[i, y_1:y_2, x_1: x_2] = self.shaped_mode(size=size, gaussian_shape=self.gaussian_shape,
                                                                     dipole=True, random_mode=self.random_mode)
                else:
                    weights[i, y_1:y_2, x_1: x_2] = self.shaped_mode(size=size, gaussian_shape=self.gaussian_shape,
                                                                     dipole=False, random_mode=self.random_mode)
                # Add constraint |W|_1 = 1
                if self.norm_weight:
                    weights[i, y_1:y_2, x_1: x_2] /= weights[i, y_1:y_2, x_1: x_2].sum()
        else:
            # Add the modes
            for i in range(self.n_variables):
                y_1, y_2, x_1, x_2 = positions[i]
                weights[i, y_1:y_2, x_1:x_2] = self.shaped_mode(size=size, gaussian_shape=self.gaussian_shape,
                                                                dipole=False, random_mode=self.random_mode)
                if self.norm_weight:
                    weights[i, y_1:y_2, x_1:x_2] /= weights[i, y_1:y_2, x_1:x_2].sum()
        return size, weights

    def generate_savar(self):

        if self.verbose:
            print("starting savar generation")

        # First generate the link_coeffs
        if self.links_coeffs is None:
            if self.verbose:
                print("Starting generation of coefficents")
            # Look for stable links
            count = 0
            while self.links_coeffs is None:
                self.links_coeffs = self.generate_links_coeff()
                count += 1
                if count == 200:
                    msg = "Impossible to find an stable links_coeff dict"
                    print(msg)
                    raise RecursionError(msg)

        # weights
        if self.mode_weights is None:
            _, self.mode_weights = self.generate_weights()

        if self.noise_weights is None:
            if self.verbose:
                print("Starting generation of weights")
            self.noise_weights = deepcopy(self.mode_weights)

        if self.n_var_ornstein is not None:
            ornstein_process = create_non_stationarity(N_var=self.n_var_ornstein,
                                                       t_sample=self.time_length,
                                                       sigma=self.ornstein_sigma,
                                                       tau=0.3)  # *variance_noise
            weights_inv = np.linalg.pinv(deepcopy(self.mode_weights).reshape(self.n_variables, -1))
            ornstein_forcing_field = (weights_inv @ ornstein_process.transpose())

        savar_dict = {
            "links_coeffs": self.links_coeffs,
            "time_length": self.time_length,
            "transient": self.transient,
            "mode_weights": self.mode_weights,
            "noise_weights": self.noise_weights,
            "noise_strength": self.noise_strength,
            "noise_variance": self.noise_variance,
            "noise_cov": self.noise_cov,
            "fast_cov": self.fast_cov,
            "latent_noise_cov": self.latent_noise_cov,
            "forcing_dict": self.forcing_dict,
            "season_dict": self.season_dict,
            "data_field": self.data_field,
            "noise_data_field": self.noise_data_field,
            "seasonal_data_field": self.seasonal_data_field,
            "forcing_data_field": self.forcing_data_field,
            "linearity": self.linearity,
            "verbose": self.verbose,
        }

        if self.verbose:
            print("starting SAVAR class")
        savar = SAVAR(**savar_dict)

        if self.n_var_ornstein is not None:
            cov = savar.generate_cov_noise_matrix()
            spatial_resolution = self.resolution[0] * self.resolution[1]
            noise_data_field = np.random.multivariate_normal(mean=np.zeros(spatial_resolution), cov=cov,
                                                             size=self.time_length + self.transient).transpose()

            noise_data_field += ornstein_forcing_field

            savar.noise_data_field = noise_data_field

        return savar

    @staticmethod
    def find_mode_positions(res: tuple = (20, 30), n_var: int = 5):

        # Loop that checks
        ny, nx = res
        size = 1
        # Loop to find the maximum size that allow the number of our modes
        while True:
            n_horiz = nx // size
            n_verti = ny // size
            total_slots = n_horiz * n_verti
            if total_slots >= n_var:  # If there is free space rise size
                size += 1
                continue
            if total_slots < n_var:  # Too large, go back and output the result
                size -= 1
                n_horiz = nx // size
                n_verti = ny // size
                break

        # We get the position of each mode
        positions = []
        for i in range(n_verti):
            for j in range(n_horiz):
                y_1 = i * size
                y_2 = i * size + size
                x_1 = j * size
                x_2 = j * size + size
                positions.append((y_1, y_2, x_1, x_2))

        return size, positions[:n_var]

    @staticmethod
    def shaped_mode(size: int, dipole: bool = False, gaussian_shape: bool = True,
                    random_mode: bool = True) -> np.ndarray:
        """
        returns either a mode with gaussian shape or squared.
        """
        frac = size // 2
        rest = size % 2
        if dipole:
            if gaussian_shape:
                if random_mode:
                    mode_1 = create_random_mode((frac, size))
                    mode_2 = create_random_mode((frac, size))
                    mode = np.zeros((size, size))
                    mode[:, :frac] = mode_1
                    mode[:, frac + rest:] = -mode_2
                    return mode
                else:
                    mode_1 = create_random_mode((frac, size), random=False)
                    mode_2 = create_random_mode((frac, size), random=False)
                    mode = np.zeros((size, size))
                    mode[:, :frac] = mode_1
                    mode[:, frac + rest:] = -mode_2
                    return mode
            else:
                mode = np.zeros((size, size))
                mode[:, :frac] = 1
                mode[:, frac + rest:] = -1
                return mode
        # No dipole
        else:
            if gaussian_shape:
                if random_mode:
                    return create_random_mode((size, size))
                else:
                    return create_random_mode((size, size), random=False)
            else:
                return np.ones((size, size))

    @staticmethod
    def check_stationarity(links, linear=True):
        """Returns stationarity according to a unit root test
        Assuming a Gaussian Vector autoregressive process
        Three conditions are necessary for stationarity of the VAR(p) model:
        - Absence of mean shifts;
        - The noise vectors are identically distributed;
        - Stability condition on Phi(t-1) coupling matrix (stabmat) of VAR(1)-version  of VAR(p).
        """

        links = deepcopy(links)
        N = len(links)
        # Check parameters
        max_lag = 0

        for j in range(N):
            for link_props in links[j]:
                var, lag = link_props[0]

                max_lag = max(max_lag, abs(lag))

        graph = np.zeros((N, N, max_lag))

        for j in range(N):
            for link_props in links[j]:
                var, lag = link_props[0]
                coeff = link_props[1]
                if abs(lag) > 0:
                    graph[j, var, abs(lag) - 1] = coeff
            if not linear:
                couplings = []
                coupling = link_props[2]
                couplings.append(coupling)

        stabmat = np.zeros((N * max_lag, N * max_lag))
        index = 0

        for i in range(0, N * max_lag, N):
            stabmat[:N, i:i + N] = graph[:, :, index]
            if index < max_lag - 1:
                stabmat[i + N:i + 2 * N, i:i + N] = np.identity(N)
            index += 1

        eig = np.linalg.eig(stabmat)[0]
        if np.all(np.abs(eig) < 1.):
            stationary = True
        else:
            stationary = False

        return stationary, np.abs(eig).max()

    @staticmethod
    def _get_random_link_strength(mean: float, std: float, strength_threshold: float = 0.2, random_sign: float = 0.5):
        """
        Returns a random number from a gaussian distribution with a given mean and std, with a minimum link strength
        and a chance that the sign is inverted.
        :param mean: mean
        :param std: standard deviation
        :param random_sign: if not None set the chances of the sign to be negative up to random_sign probability.
        :param strength_threshold: minimum link strength
        :return: the link strength
        """
        while True:
            strength = np.random.normal(mean, std)
            if strength < strength_threshold:
                continue

            if random_sign is not None:
                if np.abs(np.random.rand()) < random_sign:
                    if strength > 0:
                        return -strength
                    else:
                        return strength
            return strength


mode_configuration = {
    "size": (30, 30),
    "mu": (0, 0),
    "var": (.5, .5),
    "position": (3, 3, 3, 3),
    "plot": False,
    "Sigma": None,
    "random": True,
}
