"""
observables.py
"""
import numpy as np
from scipy.signal import correlate
from math import ceil, pi, sin

from anvil.utils import bootstrap_sample


def autocorrelation(chain):
    """Calculate the one-dimensional normalised autocorrelation function for a one-
    dimensional numpy array, given as an argument. Return positive shifts only."""
    chain_shifted = chain - chain.mean()
    auto = correlate(chain_shifted, chain_shifted, mode="same")
    t0 = auto.size // 2  # this is true for mode="same"
    return auto[t0:] / auto[t0]  # normalise and take +ve shifts


def optimal_window(integrated, mult=2.0, eps=1e-6):
    """Calculates a window length such that, when the integrated autocorrelation is
    calculated within this window, the total error is at a minimum.
    
    Notes
    -----
    See U. Wolff, Monte Carlo errors with less errors, section 3.3
    http://arXiv.org/abs/hep-lat/0306017v4
    """
    # Exponential autocorrelation
    with np.errstate(invalid="ignore", divide="ignore"):
        exponential = np.clip(
            np.nan_to_num(mult / np.log((2 * integrated + 1) / (2 * integrated - 1))),
            a_min=eps,
            a_max=None,
        )

    # Infer ensemble size. Assumes correlation mode was 'same'
    n_t = integrated.shape[-1]
    ensemble_size = n_t * 2

    # g_func is the derivative of the sum of errors wrt window size
    window = np.arange(1, n_t + 1)
    g_func = np.exp(-window / exponential) - exponential / np.sqrt(
        window * ensemble_size
    )

    # Return first occurrence of g_func changing sign
    return np.argmax((g_func[..., 1:] < 0), axis=-1)


# ------------------------------------------------------------------------------------- #
#                               Two point observables                                   #
# ------------------------------------------------------------------------------------- #


def two_point_correlator(field_ensemble, n_boot):
    """Bootstrap sample of two point connected correlation functions for the
    field ensemble."""
    return field_ensemble.boot_two_point_correlator(n_boot)


def zero_momentum_correlator(two_point_correlator):
    """Two point correlator at zero spatial momentum."""
    return 0.5 * (two_point_correlator.mean(axis=0) + two_point_correlator.mean(axis=1))


def effective_pole_mass(zero_momentum_correlator):
    r"""Effective pole mass defined by

        m_p = cosh^{-1}( (\tilde{G}(t-1) + \tilde{G}(t+1)) / (2 * \tilde{G}(t)) )

    where \tilde{G}(t) is the zero momentum correlator defined in this module.
    """
    inner_indices = np.arange(1, zero_momentum_correlator.shape[0] - 1, dtype=int)
    return np.arccosh(
        (
            zero_momentum_correlator[inner_indices - 1]
            + zero_momentum_correlator[inner_indices + 1]
        )
        / (2 * zero_momentum_correlator[inner_indices])
    )


def exponential_correlation_length(effective_pole_mass):
    """Squared exponential correlation length, defined as the inverse of the pole mass."""
    return (1 / effective_pole_mass) ** 2


def susceptibility(two_point_correlator):
    """Susceptibility defined as the first moment of the two point correlator."""
    return two_point_correlator.sum(axis=(0, 1))


def ising_energy(two_point_correlator):
    """Ising energy density, defined as the two point correlator at the minimum
    lattice spacing."""
    return (two_point_correlator[1, 0] + two_point_correlator[0, 1]) / 2


def second_moment_correlation_length(two_point_correlator, susceptibility):
    """Squared second moment correlation length, defined as the normalised second
    moment of the two point correlator."""
    L = two_point_correlator.shape[0]
    x = np.concatenate((np.arange(0, L // 2 + 1), np.arange(-L // 2 + 1, 0)))
    x1, x2 = np.meshgrid(x, x)
    x_sq = np.expand_dims((x1 ** 2 + x2 ** 2), -1)  # pick up bootstrap dimension

    mu_0 = susceptibility
    mu_2 = (x_sq * two_point_correlator).sum(axis=(0, 1))  # second moment

    return mu_2 / (4 * mu_0)  # normalisation


def low_momentum_correlation_length(two_point_correlator, susceptibility):
    """A low-momentum estimate for the squared correlation length."""
    L = two_point_correlator.shape[0]
    kernel = np.cos(2 * pi / L * np.arange(L)).reshape(L, 1, 1)

    g_tilde_00 = susceptibility
    g_tilde_10 = (kernel * two_point_correlator).sum(axis=(0, 1))

    return (g_tilde_00 / g_tilde_10 - 1) / (4 * sin(pi / L) ** 2)


def two_point_correlator_series(field_ensemble):
    """The volume-averaged two point correlator for the lowest few shifts along
    a single axes, with the ensemble dimension interpreted as a series."""
    return field_ensemble.two_point_correlator_series


def two_point_correlator_autocorr(two_point_correlator_series):
    """The autocorrelation function calculated for each shift in the two point
    correlator series."""
    return np.array(
        [
            autocorrelation(two_point_correlator_series[i])
            for i in range(two_point_correlator_series.shape[0])
        ]
    )


def two_point_correlator_integrated_autocorr(two_point_correlator_autocorr):
    r"""The integrated autocorrelation, as a function of summation window size,
    for the two point correlator series, defined as

        \tau(W) = 1/2 + \sum_{t=1}^W \Gamma(t)
    """
    return np.cumsum(two_point_correlator_autocorr, axis=-1) - 0.5


def two_point_correlator_optimal_window(two_point_correlator_integrated_autocorr):
    """The optimal value for the summation window for the two point correlator
    integrated autocorrelation."""
    return optimal_window(two_point_correlator_integrated_autocorr)


def magnetisation_series(field_ensemble):
    return field_ensemble.magnetisation_series


def _magnetisation(magnetisation_series, n_boot):
    return bootstrap_sample(magnetisation_series, n_boot)


def magnetisation(_magnetisation, training_geometry):
    return np.abs(_magnetisation).mean(axis=-1) / training_geometry.length ** 2


def magnetic_susceptibility(_magnetisation, training_geometry):
    return (
        (_magnetisation ** 2).mean(axis=-1)
    ) / training_geometry.length ** 2


def magnetic_susceptibility_v2(_magnetisation, training_geometry):
    return (
        (_magnetisation ** 2).mean(axis=-1) - np.abs(_magnetisation).mean(axis=-1) ** 2
    ) / training_geometry.length ** 2


def magnetisation_autocorr(magnetisation_series):
    return autocorrelation(magnetisation_series)


def magnetisation_integrated_autocorr(magnetisation_autocorr):
    return np.cumsum(magnetisation_autocorr, axis=-1) - 0.5


def magnetisation_optimal_window(magnetisation_integrated_autocorr):
    return optimal_window(magnetisation_integrated_autocorr)


# ------------------------------------------------------------------------------------- #
#                             Topological observables                                   #
# ------------------------------------------------------------------------------------- #


def topological_charge_series(field_ensemble):
    """The topological charge with the ensemble dimension interpreted as a series."""
    return field_ensemble.topological_charge


def _topological_charge(topological_charge_series, n_boot):
    """A bootstrap sample of topological charges, with the ensemble dimension
    not yet summed over."""
    return bootstrap_sample(topological_charge_series, n_boot)


def topological_charge(_topological_charge):
    """The ensemble-averaged topological charge."""
    return _topological_charge.mean(axis=-1)


def topological_susceptibility(_topological_charge, training_geometry):
    """Topological susceptibility, defined as the second moment of the topological
    charge divided by the lattice volume."""
    return _topological_charge.var(axis=-1) / training_geometry.length ** 2


def topological_charge_autocorr(topological_charge_series):
    """The autocorrelation function of the topological charge series."""
    return autocorrelation(topological_charge_series)


def topological_charge_integrated_autocorr(topological_charge_autocorr):
    r"""The integrated autocorrelation, as a function of summation window size,
    for the topological charge series."""
    return np.cumsum(topological_charge_autocorr) - 0.5


def topological_charge_optimal_window(topological_charge_integrated_autocorr):
    """The optimal value for the summation window for the topological charge
    integrated autocorrelation."""
    return optimal_window(topological_charge_integrated_autocorr)
