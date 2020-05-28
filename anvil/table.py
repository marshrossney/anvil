"""
table.py

Module containing all table actions

"""
import numpy as np
import pandas as pd

from reportengine.table import table


@table
def table_two_point_scalars(ising_energy, susceptibility):
    """Table of the ising observables, with mean and standard deviation taken
    across boostrap samples
    """
    res = [
        [ising_energy.mean(), ising_energy.std()],
        [susceptibility.mean(), susceptibility.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Ising energy", "susceptibility"],
    )
    return df


@table
def table_correlation_length(
    exponential_correlation_length, low_momentum_correlation_length
):
    # Weighted average of positive shifts, ignoring first point
    T = exponential_correlation_length.shape[0]
    ecl_values = exponential_correlation_length[1 : T // 2 + 1].mean(axis=-1)
    ecl_errors = exponential_correlation_length[1 : T // 2 + 1].std(axis=-1)
    weights = 1 / ecl_errors
    ecl_mean = np.sum(ecl_values * weights) / np.sum(weights)
    ecl_error = 1 / np.sum(weights)  # standard error on the mean

    res = [
        [ecl_mean, ecl_error],
        [low_momentum_correlation_length.mean(), low_momentum_correlation_length.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Exp corr length", "Low mom corr length"],
    )
    return df


@table
def table_zero_momentum_correlator(zero_momentum_correlator, training_geometry):
    """Table of zero_momentum_correlator, with mean and standard deviation
    from bootstrap
    """
    means = zero_momentum_correlator.mean(axis=-1)[:, np.newaxis]
    stds = zero_momentum_correlator.std(axis=-1)[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)

    df = pd.DataFrame(
        data,
        columns=["Mean", "Standard deviation"],
        index=range(training_geometry.length),
    )
    return df


@table
def table_effective_pole_mass(effective_pole_mass, training_geometry):
    """Table of effective_pole_mass, with mean and standard deviation
    from bootstrap
    """
    means = effective_pole_mass.mean(axis=-1)[:, np.newaxis]
    stds = effective_pole_mass.std(axis=-1)[:, np.newaxis]

    data = np.concatenate((means, stds), axis=1)
    df = pd.DataFrame(
        data,
        columns=["Mean", "Standard deviation"],
        index=range(1, training_geometry.length - 1),
    )
    return df


@table
def table_two_point_correlator(training_geometry, two_point_correlator):
    """For each x and t, tabulate the mean and standard deviation of the two
    point function, estimated from bootstrap sample
    """
    corr = []
    index = []
    means = two_point_correlator.mean(axis=-1)
    stds = two_point_correlator.std(axis=-1)

    for i in range(training_geometry.length):
        for j in range(training_geometry.length):
            corr.append([float(means[i, j]), float(stds[i, j])])
            index.append((i, j))
    df = pd.DataFrame(corr, columns=["Mean", "Standard deviation"], index=index)
    return df


@table
def table_two_point_correlator_autocorr(two_point_correlator_autocorr):
    """Tabulate the autocorrelation of the two point function"""
    df = pd.DataFrame(
        two_point_correlator_autocorr,
        columns=["Autocorrelation"],
        index=range(len(two_point_correlator_autocorr)),
    )
    return df


@table
def table_topological_observables(topological_charge, topological_susceptibility):
    res = [
        [topological_charge.mean(), topological_charge.std()],
        [topological_susceptibility.mean(), topological_susceptibility.std()],
    ]
    df = pd.DataFrame(
        res,
        columns=["Mean", "Standard deviation"],
        index=["Topological charge", "Topological susceptibility"],
    )
    return df
