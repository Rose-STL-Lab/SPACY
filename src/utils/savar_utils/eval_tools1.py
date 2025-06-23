# This file containts the functions of the experiments

# Import internals
from savar.savar import SAVAR
from savar.dim_methods import get_varimax_loadings_standard as varimax
from savar.c_functions import find_permutation, create_graph
# from savar.model_generator import generate_random_coeff, generate_savar_model
from savar.model_generator import SavarGenerator
from savar.functions import cg_to_est_phi

# Tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
# from tigramite.independence_tests import ParCorr
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import LinearMediation, Prediction

# Import externals
import numpy as np
from copy import deepcopy
import itertools as it

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

"""
Evaluation metrics
------------------
Weights:
- spatial correlation of weights
- correlation of signals
Phi
- Recall/Precision
"""

# WEIGHTS
# Spatial correlation of weights and signal
# Data must be introduces as K times L or K times T.
from savar.c_functions import compare_weights as compare_weights_signal

# Cg_to_est_phi allows to compare phi
from savar.functions import compare_phi, cg_to_est_phi


class DmMethod:
    """
    This method handles the different methods evaluated. Is the input of the evaluation class.
    Performs the different experimetns

    """

    def __init__(self, savar: SAVAR, tau_max: int = None, max_comps: int = None,
                 significance: str = "analytic", ind_test: str = "ParCorr", pc_alpha: float = 0.2,
                 parents_alpha: float = 0.05, correct_permutation: bool = True, perform_analysis: bool = False,
                 verbose: bool = True):

        # Input objects
        self.savar = savar
        print("correct_permutation",correct_permutation)
        # Checks
        if self.savar.data_field is None:
            raise Exception("saver is empty did you run the method for create data?")

        # Tigramite configuration
        if self.savar.linearity == "linear":
            _, max_lag = create_graph(self.savar.links_coeffs, return_lag=True)
        else:  # Get the max lag for non linear
            max_lag = max(abs(lag)
                          for (_, lag), _, _ in it.chain.from_iterable(self.savar.links_coeffs.values()))
        if tau_max is None:
            self.tau_max = max_lag
        else:
            if tau_max < max_lag:
                raise Exception("Tau max is smaller than the true tau_max")
            self.tau_max = tau_max
        self.significance = significance
        self.ind_test = ind_test
        self.pc_alpha = pc_alpha
        self.parents_alpha = parents_alpha

        # Set configuration
        if max_comps is None:  # Assign the true components
            self.max_comps = self.savar.n_vars
#             self.max_comps = self.savar.n_variables
        else:
            self.max_comps = max_comps
            ## Check that components are <= savar.variables
            if max_comps < self.savar.n_vars:
                print("Savar Models has {} components while max_comps = {}, the latter should be \
                                Equal or bigger".format(savar.n_vars, self.max_comps))
                raise Exception("Savar Models has {0} components while max_comps = {1}, the latter should be"
                                "equal or biggerto {0}".format(savar.n_vars, self.max_comps))
        self.correct_permutation = correct_permutation
        self.verbose = verbose

        # Extract savar elements
        self.data_field = deepcopy(self.savar.data_field.transpose())  # L times T

        # Emtpy elements
        # Savar
        self.phi = {}
        self.weights = {}
        self.weights_inv = {}
        self.signal = {}
        self.x_prediction = {}
        self.y_prediction = {}

        # Tigramite
        self.pcmci = {}
        self.grid_pcmci = {}
        self.tg_results = {}
        self.tg_grid_results = {}
        self.parents_predict = {}
        self.grid_phi = {}

        # Others
        self.varimax_out = None
        self.pca_out = None
        self.permutation_dict = {}
        self.is_permuted = False
        self.is_grid_done = False

        # If perform analisis = True does all the steps
        if perform_analysis:
            if verbose:
                print("Starting DR Methods")
            self.perform_dm()
            self.get_pcmci_results()
            self.get_phi_and_predict()
            if verbose:
                print("DR Methods finished")

    def perform_dm(self) -> None:
        """
        Perform a dimensionality reduction method, save both varimax and pca.
        :return:
        """

        #### VARIMAX ####
        # Input varimax: T times L
        print("datafield:",self.data_field.transpose().shape)
        self.varimax_out = varimax(self.data_field, max_comps=self.max_comps, verbosity=self.verbose)
        
        self.weights["varimax"] = self.varimax_out["weights"].transpose()  # K times L
        print("varimax", self.weights["varimax"].shape)

        # signal:  K times L @ L times T -> K times T (signal)
        print("shapes",self.data_field.shape, self.weights["varimax"].shape)
        self.signal["varimax"] = self.weights["varimax"] @ self.data_field.transpose()
        print("signal1", self.signal["varimax"].shape)

        # If correct permutation we fix the permutation
        if self.correct_permutation:
            savar = deepcopy(self.savar)
#             print(savar.mode_weights.shape)
            print(savar.data_field.shape,savar.mode_weights.shape)
            
            savar_signal = savar.data_field.transpose() @ savar.mode_weights.reshape(savar.n_vars, -1).transpose()
#             savar_signal = savar.mode_weights.reshape(savar.n_vars, -1).transpose()@savar.data_field
            self.permutation_dict["varimax"] = find_permutation(savar_signal, self.signal["varimax"])
            idx_permutation = [self.permutation_dict["varimax"][i] for i in range(savar.n_vars)]

            # Correct weights and signal
            self.weights["varimax"] = self.weights["varimax"][idx_permutation]
            self.signal["varimax"] = self.weights["varimax"] @ self.data_field.transpose()

            self.is_permuted = True

        self.weights_inv["varimax"] = np.linalg.pinv(self.weights["varimax"])  # L times K

        ### PCA ###
        self.pca_out = PCA(n_components=self.max_comps)
        self.pca_out = PCA(n_components=self.max_comps)
        # Input PCA: T times L
        self.pca_out.fit(self.data_field)
        self.weights["pca"] = self.pca_out.components_

        # signal =   K times L @ L times T -> K times T (signal)
        print("pca", self.weights["pca"].shape)
        self.signal["pca"] = self.weights["pca"] @ self.data_field.transpose()
        print("signal2", self.signal["pca"].shape)

        # If correct permutation we fix the permutation
        if self.correct_permutation:
            savar = deepcopy(self.savar)
            savar_signal = savar.data_field.transpose() @ savar.mode_weights.reshape(savar.n_vars, -1).transpose()
            print("signals: ", savar_signal.shape, self.signal["pca"].shape)
            self.permutation_dict["pca"] = find_permutation(savar_signal, self.signal["pca"])
            idx_permutation = [self.permutation_dict["pca"][i] for i in range(savar.n_vars)]

            # New weights and signal
            self.weights["pca"] = self.weights["pca"][idx_permutation]
            self.signal["pca"] = self.weights["pca"] @ self.data_field.transpose()

        self.weights_inv["pca"] = np.linalg.pinv(self.weights["pca"])

    def get_pcmci_results(self) -> None:
        """
        Performs PCMCI
        :return: None
        """
        if self.significance == "analytic" and self.ind_test == "ParCorr":
            ind_test = ParCorr(significance=self.significance)
        else:
            raise ("Only ParrCorr test implemented your option: {} not yet implemented".format(self.ind_test))

        # Varimax
        self.pcmci["varimax_pcmci"] = PCMCI(
            dataframe=pp.DataFrame(self.signal["varimax"].transpose()),  # Input data for PCMCI T times K
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )
        self.pcmci["varimax_corr"] = deepcopy(self.pcmci["varimax_pcmci"])
        self.tg_results["varimax_pcmci"] = self.pcmci["varimax_pcmci"].run_pcmciplus(tau_min=1, tau_max=self.tau_max,
                                                                         pc_alpha=self.pc_alpha)
        self.tg_results["varimax_corr"] = self.pcmci["varimax_corr"].get_lagged_dependencies(selected_links=None,
                                                                                                  tau_min=1,
                                                                                                  tau_max=self.tau_max,
                                                                                                  val_only=False)
        # PCA
        ind_test = ParCorr(significance=self.significance)
        self.pcmci["pca_pcmci"] = PCMCI(
            dataframe=pp.DataFrame(self.signal["pca"].transpose()),  # Input data for PCMCI T times K
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )
        self.pcmci["pca_corr"] = deepcopy(self.pcmci["pca_pcmci"])
        self.tg_results["pca_pcmci"] = self.pcmci["pca_pcmci"].run_pcmciplus(tau_min=1, tau_max=self.tau_max,
                                                                         pc_alpha=self.pc_alpha)
        self.tg_results["pca_corr"] = self.pcmci["pca_corr"].get_lagged_dependencies(selected_links=None,
                                                                                                  tau_min=1,
                                                                                                  tau_max=self.tau_max,
                                                                                                  val_only=False)

    def get_phi_and_predict(self):
        """
        here are 4 Phi, depending on the method.
        TODO: Prediction only implemented for pcmci methods.
        :return:
        """
        # Method 1: Varimax Corr
        corr_results_var_corr = deepcopy(self.tg_results["varimax_corr"])
        # Used to convert linear OLS result from pearson coefficient
#         variance_vars = self.pcmci["varimax_corr"].dataframe.values.std(axis=0)
        variance_vars = np.std(self.pcmci["varimax_corr"].dataframe.values[0], axis = 0)
        # Get Phi from val_matrix
        Phi = corr_results_var_corr['val_matrix']

        # If p_value not enought set it to 0
#         print(corr_results_var_corr['p_matrix'].shape, self.pc_alpha, Phi.shape)
#         Phi[[corr_results_var_corr['p_matrix'] > self.pc_alpha]] = 0
        Phi[corr_results_var_corr['p_matrix'] > self.pc_alpha] = 0

        # Now we do the coefficient by Val_matrix[i, j, tau]*std(j)/std(i)
        Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]

        self.phi["varimax_corr"] = np.moveaxis(deepcopy(Phi), 2, 0)
        np.fill_diagonal(self.phi["varimax_corr"][0, ...], 1)  # Fill the diagonal of tau 0 with ones

        # Method 2: Varimax PCMCI
        # Get parents
#         self.parents_predict["varimax_pcmci"] = self.pcmci["varimax_pcmci"].return_significant_parents(
#             pq_matrix=self.tg_results["varimax_pcmci"]["p_matrix"],
#             val_matrix=self.tg_results["varimax_pcmci"]["val_matrix"],
#             alpha_level=self.parents_alpha,
#             include_lagzero_parents=False,
#         )["parents"]

        self.parents_predict["varimax_pcmci"] = self.pcmci["varimax_pcmci"].return_parents_dict(
            graph=self.tg_results["varimax_pcmci"]["graph"],
            val_matrix=self.tg_results["varimax_pcmci"]["val_matrix"],
            include_lagzero_parents=False)

        # Get model and phi
        dataframe = deepcopy(self.pcmci["varimax_pcmci"].dataframe)
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=self.parents_predict["varimax_pcmci"], tau_max=self.tau_max)
        self.phi["varimax_pcmci"] = med.phi

        # Prediction (only for pcmci methods)
#         print(dataframe.values)
        T, N = dataframe.values[0].shape

        pred = Prediction(dataframe=dataframe,
                          cond_ind_test=None,
                          prediction_model=LinearRegression(),
                          train_indices=range(T),
                          test_indices=range(T),
                          data_transform=None,
                          verbosity=self.verbose
                          )

        target_vars = range(N)
        predict_matrix = np.zeros((N, T - self.tau_max))
        for var in target_vars:
            if len(self.parents_predict["varimax_pcmci"][var]) > 0:
                # Fit for all the variables
                pred.fit(target_predictors=self.parents_predict["varimax_pcmci"],
                         selected_targets=[var],  # Requires a list
                         tau_max=self.tau_max,
                         )

                predict_matrix[var, :] = pred.predict(var, new_data=None)

        self.x_prediction["pca_corr"] = predict_matrix  # K times T

        # Method 3: PCA Corr
        corr_results_pca_corr = deepcopy(self.tg_results["pca_corr"])
        # Used to convert linear OLS result from pearson coefficient
#         variance_vars = self.pcmci["pca_corr"].dataframe.values.std(axis=0)
        variance_vars = np.std(self.pcmci["pca_corr"].dataframe.values[0], axis = 0)


        # Get Phi from val_matrix
        Phi = corr_results_pca_corr['val_matrix']

        # If p_value not enough set it to 0
        Phi[corr_results_pca_corr['p_matrix'] > self.pc_alpha] = 0

        # Now we do the coefficient by Val_matrix[i, j, tau]*std(j)/std(i)
        Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]

        self.phi["pca_corr"] = np.moveaxis(deepcopy(Phi), 2, 0)  # Phi is in the other cases [tau, i, j],
        np.fill_diagonal(self.phi["pca_corr"][0, ...], 1)  # Fill the diagonal of tau 0 with ones

        # Method 4
#         self.parents_predict["pca_pcmci"] = self.pcmci["pca_pcmci"].return_significant_parents(
#             pq_matrix=self.tg_results["pca_pcmci"]["p_matrix"],
#             val_matrix=self.tg_results["pca_pcmci"]["val_matrix"],
#             alpha_level=self.parents_alpha,
#             include_lagzero_parents=False,
#         )["parents"]
        self.parents_predict["pca_pcmci"] = self.pcmci["pca_pcmci"].return_parents_dict(
            graph=self.tg_results["pca_pcmci"]["p_matrix"],
            val_matrix=self.tg_results["pca_pcmci"]["val_matrix"],
            include_lagzero_parents=False)
        dataframe = deepcopy(self.pcmci["pca_pcmci"].dataframe)

        # Get model and phi
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=self.parents_predict["pca_pcmci"], tau_max=self.tau_max)
        self.phi["pca_pcmci"] = med.phi

        # Prediction (only for pcmci methods)
        T, N = dataframe.values[0].shape

        pred = Prediction(dataframe=dataframe,
                          cond_ind_test=None,
                          prediction_model=LinearRegression(),
                          train_indices=range(T),
                          test_indices=range(T),
                          data_transform=None,
                          verbosity=self.verbose
                          )

        target_vars = range(N)
        predict_matrix = np.zeros((N, T - self.tau_max))
        for var in target_vars:
            if len(self.parents_predict["pca_pcmci"][var]) > 0:
                # Fit for all the variables
                pred.fit(target_predictors=self.parents_predict["pca_pcmci"],
                         selected_targets=[var],  # Requires a list
                         tau_max=self.tau_max,
                         )

                predict_matrix[var, :] = pred.predict(var, new_data=None)

        self.x_prediction["pca_pcmci"] = predict_matrix  # K times T

    def get_grid_phi(self) -> None:
        """
        Performs PCMCI and computes the phi at grid level
        Performs parCorr and computes the phi at grid level
        :return: None
        """
        if self.is_grid_done:
            print("Grid level already performed")
            return None

        if self.verbose:
            print("Starting pcmci at grid level")

        if self.significance == "analytic" and self.ind_test == "ParCorr":
            ind_test = ParCorr(significance=self.significance)
        else:
            raise ("Only ParrCorr test implemented your option: {} not yet implemented".format(self.ind_test))
        dataframe = pp.DataFrame(self.savar.data_field.transpose())  # Input data for PCMCI T times K

        #### PCMCI ####
        self.grid_pcmci["pcmci"] = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )
        print("attributes",dir(self.grid_pcmci["pcmci"]))
        self.tg_grid_results["pcmci"] = self.grid_pcmci["pcmci"].run_pcmciplus(tau_min=1, tau_max=self.tau_max,
                                                                               pc_alpha=self.pc_alpha)
#         self.parents_predict["pcmci"] = self.grid_pcmci["pcmci"].return_significant_parents(
#             pq_matrix=self.tg_grid_results["pcmci"]["p_matrix"],
#             val_matrix=self.tg_grid_results["pcmci"]["val_matrix"],
#             alpha_level=self.parents_alpha,
#             include_lagzero_parents=False,
#         )["parents"]
        self.parents_predict["pcmci"] = self.grid_pcmci["pcmci"].return_parents_dict(
            graph=self.tg_grid_results["pcmci"]["graph"],
            val_matrix=self.tg_grid_results["pcmci"]["val_matrix"],
            include_lagzero_parents=False)
        print("val_matrix",self.tg_grid_results["pcmci"]["val_matrix"].shape)


        # Get grid phi
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=self.parents_predict["pcmci"], tau_max=self.tau_max)
        self.grid_phi["pcmci"] = med.phi
        print("PCMCI_phi", med.phi.shape)

        #### Correlation ####

        self.grid_pcmci["corr"] = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ind_test,
            verbosity=self.verbose,
        )

        self.tg_grid_results["corr"] = self.grid_pcmci["corr"].get_lagged_dependencies(selected_links=None,
                                                                                             tau_min=1,
                                                                                             tau_max=self.tau_max,
                                                                                             val_only=False)

        corr_grd_results = deepcopy(self.tg_grid_results["corr"])
        variance_vars = np.std(self.grid_pcmci["corr"].dataframe.values[0], axis = 0)

#         variance_vars = self.grid_pcmci["corr"].dataframe.values.std(axis=0)

        # Get Phi from val_matrix
        Phi = corr_grd_results['val_matrix']

        # If p_value not enought set it to 0
        Phi[corr_grd_results['p_matrix'] > self.pc_alpha] = 0

        # Now we do the coefficient by Val_matrix[i, j, tau]*std(j)/std(i)
        Phi = (Phi * variance_vars[:, None]) / variance_vars[:, None, None]

        self.grid_phi["corr"] = np.moveaxis(deepcopy(Phi), 2, 0)
        np.fill_diagonal(self.grid_phi["corr"][0, ...], 1)  # Fill the diagonal of tau 0 with ones

        # Set it so its not performed again
        self.is_grid_done = True


# We will use isa_pcmci to get Phi before any iteration.
class Evaluation:
    """
    This class is used to output the evaluation metrics of the experiments
    """

    def __init__(self, dm_object: DmMethod, grid_threshold: float = None, grid_threshold_per: float = 95, verbose: bool = True,
                 methods=("varimax_corr", "varimax_pcmci", "pca_corr", "pca_pcmci"),
                 grid_methods=("pcmci", "corr", "varimax_pcmci_w", "varimax_corr_w",
                               "pca_pcmci_w", "pca_corr_w")):
        # inputs
        self.dm_object = dm_object

        # Extract dm_object
        self.n_vars = self.dm_object.savar.n_vars
        self.tau_max = self.dm_object.tau_max
        self.dm_phi = self.dm_object.phi
        self.dm_cg = deepcopy(self.dm_phi)
        for key in self.dm_cg:
            self.dm_cg[key][np.abs(self.dm_phi[key]) > 0] = 1
        self.dm_phi = self.dm_object.phi
        self.dm_weights = self.dm_object.weights

        # savar
        self.savar_phi = cg_to_est_phi(self.dm_object.savar.links_coeffs, tau_max=self.dm_object.tau_max)
        self.savar_weights = self.dm_object.savar.mode_weights.reshape(self.n_vars, -1)
        self.savar_cg = deepcopy(self.savar_phi)
        self.savar_cg[np.abs(self.savar_cg) > 0] = 1

        # Other
        self.verbose = verbose
        self.grid_threshold = grid_threshold
        self.grid_threshold_per = grid_threshold_per
        self.metrics = {metric: {} for metric in methods + grid_methods}
        self.methods = methods
        self.grid_methods = grid_methods

        # Empty attributes
        self.cg_conf_matrix = {}
        self.savar_grid_phi = None
        self.grid_phi = {method: {} for method in grid_methods}
        self.savar_grid_cg = None
        self.grid_cg = {method: {} for method in grid_methods}
        self.dm_grid_cg = None
        self.cg_grid_conf_matrix = {}
        self.dm_grid_phi = None

    def _obtain_individual_metrics(self, method):

        # MSE and RMAE
        savar_phi = self.savar_phi[1:, ...]
        idx = np.nonzero(savar_phi)  # Non_zero elements of True phi
        dm_phi = deepcopy(self.dm_object.phi[method][1:, ...])
        # For non-zero elements of savar phi = (|Phi-\tilde(Phi)|)/|Phi|
        self.metrics[method]["mse"] = np.square(savar_phi[idx] - dm_phi[idx]).mean()
        self.metrics[method]["rmae"] = (np.abs(savar_phi[idx]-dm_phi[idx])/np.abs(savar_phi[idx])).mean()

        # Precision and Recall
        savar_cg = deepcopy(self.savar_cg[1:, ...].flatten())
        dm_cg = self.dm_cg[method][1:, ...].flatten()
#         print(savar_cg, dm_cg)
        self.cg_conf_matrix[method] = confusion_matrix(savar_cg, dm_cg, labels=(0, 1))
        tn, fp, fn, tp = self.cg_conf_matrix[method].ravel()

        if (tp + fp) != 0:
            self.metrics[method]["precision" ] = tp / (tp + fp)
        else:
            self.metrics[method]["precision"] = 0
        if tp / (tp + fn) != 0:
            self.metrics[method]["recall"] = tp / (tp + fn)
        else:
            self.metrics[method]["recall"] = 0
        if tn / (tn + fp) != 0:
            self.metrics[method]["tnr"] = tn / (tn + fp)
        else:
            self.metrics[method]["tnr"] = 0
        self.metrics[method]["tpr"] = self.metrics[method]["recall"]
           

        # Weights
        if method in ("varimax_pcmci", "varimax_corr"):
            method_dm = "varimax"
        elif method in ("pca_pcmci", "pca_corr"):
            method_dm = "pca"
        else:
            raise Exception("Method {} not correct".format(method))
        N = self.dm_object.savar.n_vars
        lag,nx,ny = self.dm_object.savar.mode_weights.shape
        print("weights:", self.dm_object.weights[method_dm].shape, self.dm_object.savar.mode_weights.shape)
#         corr_weights = np.array([np.corrcoef(self.dm_object.weights[method_dm][i, ...],
#                                              self.dm_object.savar.mode_weights[i, ...].flatten())[0, 1]
#                                  for i in range(N)])
        corr_weights = np.array([np.corrcoef(self.dm_object.weights[method_dm][i, ...],
                                             self.dm_object.savar.mode_weights.reshape(lag, nx*ny)[i, ...])[0, 1]
                                 for i in range(N)])



        self.metrics[method]["corr_weights"] = np.abs(corr_weights)

        # Signal
        # (K times T) = K times L @ L times T
        savar_signal = self.dm_object.savar.mode_weights.reshape(N, -1) @ self.dm_object.savar.data_field
        corr_signal = np.array([np.corrcoef(self.dm_object.signal[method_dm][i, ...], savar_signal[i, ...])[0, 1]
                                for i in range(N)])
        self.metrics[method]["corr_signal"] = np.abs(corr_signal)

    def _obtain_grid_metrics(self):

        dm_grid_methods_dict = {
        "varimax_pcmci_w" : ("varimax_pcmci", "varimax"),
        "varimax_corr_w" : ("varimax_corr", "varimax"),
        "pca_pcmci_w" : ("pca_pcmci", "pca"),
        "pca_corr_w": ("pca_corr", "pca"),
        }

        self.dm_object.get_grid_phi()
        # Grid_phi = W^+ Phi W
        self.savar_grid_phi = self.compute_grid_phi(self.savar_phi, self.savar_weights)
        idx = np.nonzero(self.savar_grid_phi)  # Non_zero elements of True phi

        for method in self.grid_methods:
            if method in ("pcmci", "corr"):

                # MSE and RMAR
                self.grid_phi[method] = deepcopy(self.dm_object.grid_phi[method][1:, ...])
                dm_grid_phi = deepcopy(self.grid_phi[method])
                savar_grid = deepcopy(self.savar_grid_phi)
                # For non-zero elements of savar phi = (|Phi-\tilde(Phi)|)/|Phi|
                print("Shapes1 "+method,dm_grid_phi.shape)
                print("Shapes2", savar_grid.shape, dm_grid_phi.shape)
                self.metrics[method]["grid_mse"] = np.square(savar_grid[idx] - dm_grid_phi[idx]).mean()
                self.metrics[method]["grid_rmae"] = (np.abs(savar_grid[idx] - dm_grid_phi[idx]) / np.abs(savar_grid[idx])).mean()

                # CG metrics (precisions and Recall)
                self.savar_grid_cg = deepcopy(self.savar_grid_phi)
                self.savar_grid_cg[np.abs(self.savar_grid_cg) > 0] = 1
                self.grid_cg[method] = deepcopy(self.grid_phi[method])
                self.grid_cg[method][np.abs(self.grid_cg[method]) > 0] = 1

                savar_grid_cg = self.savar_grid_cg.flatten()
                grid_cg = self.grid_cg[method].flatten()
                print("shapes", self.savar_grid_cg.shape, self.grid_cg[method].shape)
                self.cg_grid_conf_matrix[method] = confusion_matrix(savar_grid_cg, grid_cg, labels=(0, 1))
                tn, fp, fn, tp = self.cg_grid_conf_matrix[method].ravel()

                self.metrics[method]["grid_precision"] = tp / (tp + fp)
                self.metrics[method]["grid_recall"] = tp / (tp + fn)
                self.metrics[method]["grid_tpr"] = self.metrics[method]["grid_recall"]
                self.metrics[method]["grid_tnr"] = tn / (tn + fp)

            if method in ("varimax_pcmci_w", "varimax_corr_w",
                               "pca_pcmci_w", "pca_corr_w"):

                latent_method = dm_grid_methods_dict[method][0]  # e.j: varimax_pcmci
                dm_method = dm_grid_methods_dict[method][1]  # e.j: varimax

                self.grid_phi[method] = self.compute_grid_phi(
                    self.dm_object.phi[latent_method],
                    self.dm_weights[dm_method])

                dm_grid_phi = deepcopy(self.grid_phi[method])
                savar_grid = deepcopy(self.savar_grid_phi)

                # For non-zero elements of savar phi = (|Phi-\tilde(Phi)|)/|Phi|
                self.metrics[method]["grid_mse"] = np.square(savar_grid[idx] - dm_grid_phi[idx]).mean()
                self.metrics[method]["grid_rmae"] = (np.abs(savar_grid[idx] - dm_grid_phi[idx])
                                                     / np.abs(savar_grid[idx])).mean()

                self.savar_grid_cg = deepcopy(self.savar_grid_phi)
                self.savar_grid_cg[np.abs(self.savar_grid_cg) > 0] = 1

                self.grid_cg[method] = deepcopy(self.grid_phi[method])
                # Apply a threshold
                if self.grid_threshold is not None:
                    grid_threshold = self.grid_threshold
                else:
                    grid_threshold = np.percentile(np.abs(self.grid_cg[method]), self.grid_threshold_per)
                print("printing percent threshold", grid_threshold)
                self.grid_cg[method][np.abs(self.grid_cg[method]) <= grid_threshold] = 0
                self.grid_cg[method][np.abs(self.grid_cg[method]) > 0] = 1

                savar_grid_cg = self.savar_grid_cg.flatten()
                grid_cg = self.grid_cg[method].flatten()

                self.cg_grid_conf_matrix[method] = confusion_matrix(savar_grid_cg, grid_cg, labels=(0, 1))
                tn, fp, fn, tp = self.cg_grid_conf_matrix[method].ravel()

                self.metrics[method]["grid_precision"] = tp / (tp + fp)
                self.metrics[method]["grid_recall"] = tp / (tp + fn)
                self.metrics[method]["grid_tpr"] = self.metrics[method]["grid_recall"]
                self.metrics[method]["grid_tnr"] = tn / (tn + fp)

    def obtain_score_metrics(self, perform_grid=True):
        """
        Computes the following metrics for the 4 methods implemented
        If perfrom_grid, then also performs the metrics at grid level
        :return:
        """
        for method in self.methods:
            self._obtain_individual_metrics(method)

        if perform_grid:
            self._obtain_grid_metrics()

    @staticmethod
    def compute_grid_phi(phi, weights):
        """ Remove time 0 in tau and computes the grid version
        :param phi: Phi at mode level
        :param weights: weights of shape K x L"""
        # Estimated Grid_Phi
        phi = phi[1:, ...]
        weights_inv = np.linalg.pinv(weights)
        grid_phi = weights_inv[None, ...] @ phi @ weights[None, ...]
        return weights_inv[None, ...] @ phi @ weights[None, ...]


if __name__ == "__main__":

#     savar = generate_savar_model(3, tau_max=3, resolution=(10, 20), gaussian_shape=False)
    savar = SavarGenerator.generate_savar_model(3, tau_max=3, resolution=(10, 20), gaussian_shape=False)
    savar.create_linear_savar_data()

    dm_meth = DmMethod(savar, perform_analysis=False)
    dm_meth.perform_dm()
    dm_meth.get_pcmci_results()
    dm_meth.get_phi_and_predict()

    eval = Evaluation(dm_meth)
    eval.obtain_score_metrics(perform_grid=True)
    print(eval.metrics)

    if False:
        ##################################
        # Savar imports
        import old_spatial_models as models
        from functions import create_random_mode, check_stability, compare_phi, compare_scaled, cg_to_est_phi
        from c_dim_methods import get_varimax_loadings_standard as varimax

        # Tigramite imports
        from tigramite.independence_tests import ParCorr
        import tigramite.data_processing as pp
        from tigramite.pcmci import PCMCI

        import numpy as np

        N = 3
        nx = 90  # Each component is 30x30
        ny = 30
        T = 4000  # Time

        spatial_weight = 0.5
        max_comp = 3
        tau_max = 2

        # We can create the modes with it.
        noise_weights = np.zeros((N, nx, ny))
        # noise_weights[0, :30, :] = create_random_mode((30, 30), random = False)
        # noise_weights[1, 30:60, :] = create_random_mode((30, 30), random = False)
        # noise_weights[2, 60:, :] = create_random_mode((30, 30), random = False)

        noise_weights[0, 5:25, :] = spatial_weight
        noise_weights[1, 35:55, :] = 0.2 * spatial_weight
        noise_weights[2, 65:85, :] = 0.2 * spatial_weight

        # We can use the same
        mode_weights = noise_weights

        links_coeffs = {  # TODO: Change lags to be negative
            0: [((0, -1), 0.8)],
            1: [((1, -1), 0.2), ((0, -2), 0.8)],
            2: [((2, -1), 0.2), ((0, -2), 0.8)]
        }

        check_stability(links_coeffs)

        real_phi = cg_to_est_phi(links_coeffs, tau_max=tau_max)

        spatial_covariance = 1  # 2, 1 is good
        variance_noise = 1

        savar = models.savarModel(
            links_coeffs=links_coeffs,
            mode_weights=mode_weights,
            nx=nx,
            ny=ny,
            T=T,
            spatial_covariance=spatial_covariance,
            variance_noise=variance_noise,
            covariance_noise_method="weights_transposed",  # geometric_mean, equal_noise, weights_transposed
            noise_weights=noise_weights,
            transient=200,
            season=None,
            n_vars=N,
            verbose=False
        )
        savar.create_linear_savar_data()

        dm_method = dmMethod(savar, max_comps=6)
        dm_method.perform_dm()
        dm_method.get_pcmci_results()
        dm_method.get_phi_and_predict()

        evaluator = evaluation(dm_method)
        evaluator.obtain_score_metrics()
        print(evaluator.metrics)
