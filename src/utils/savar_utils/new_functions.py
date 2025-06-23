import itertools as it
import numpy as np


def dict_to_matrix(links_coeffs, default=0):
    """
    Maps to the coefficient matrix. Without time
    :param links_coeffs:
    :param default:
    :return: a matrix coefficient of [j, i, \tau-1] where a link is i -> j at \tau
    """
    tau_max = max(abs(lag)
                  for (_, lag), _ in it.chain.from_iterable(links_coeffs.values()))

    n_vars = len(links_coeffs)

    graph = np.ones((n_vars, n_vars, tau_max))
    graph *= default

    for j, values in links_coeffs.items():
        for (i, tau), coeff in values:
            graph[j, i, abs(tau) - 1] = coeff

    return graph
