"""
plot.py

module containing all actions for plotting observables

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from reportengine.figure import figure, figuregen
from reportengine import collect


def field_component(i, x_base, phi_model, base_neg, model_neg, phi_target=None):
    fig, ax = plt.subplots()

    ax.hist(x_base, bins=50, density=True, histtype="step", label="base")
    ax.hist(phi_model, bins=50, density=True, histtype="step", label="model, full")
    ax.hist(base_neg, bins=50, density=True, histtype="step", label="model, $M_{base} < 0$")
    ax.hist(model_neg, bins=50, density=True, histtype="step", label="model, $M_{mod} < 0$")
    if phi_target is not None:
        ax.plot(*phi_target, label="target")

    ax.set_title(f"Coordinate {i}")
    ax.legend()
    fig.tight_layout()
    return fig


def field_components(loaded_model, base_dist, target_dist, lattice_size):
    """Plot the distributions of base coordinates 'x' and output coordinates 'phi' and,
    if known, plot the pdf of the target distribution."""
    sample_size = 10000

    # Generate a large sample from the base distribution and pass it through the trained model
    with torch.no_grad():
        x_base, base_log_density = base_dist(sample_size)
        sign = x_base.sum(dim=1).sign()
        neg = (sign < 0).nonzero().squeeze() 
        phi_model, model_log_density = loaded_model(x_base, base_log_density, neg)

    base_neg = phi_model[neg]
        
    sign = phi_model.sum(dim=1).sign()
    neg = (sign < 0).nonzero().squeeze()
    model_neg = phi_model[neg]

    # Convert to shape (n_coords, sample_size * lattice_size)
    x_base = x_base.reshape(sample_size * lattice_size, -1).transpose(0, 1)
    phi_model = phi_model.reshape(sample_size * lattice_size, -1).transpose(0, 1)

    base_neg = base_neg.reshape(1, -1)
    model_neg = model_neg.reshape(1, -1)

    # Include target density if known
    if hasattr(target_dist, "pdf"):
        phi_target = target_dist.pdf
    else:
        phi_target = [None for _ in range(x_base.shape[0])]

    for i in range(x_base.shape[0]):
        yield field_component(i, x_base[i], phi_model[i], base_neg[i], model_neg[i], phi_target[i])


_plot_field_components = collect("field_components", ("training_context",))


def example_configs(loaded_model, base_dist, training_geometry):
    sample_size = 10

    # Generate a large sample from the base distribution and pass it through the trained model
    with torch.no_grad():
        x_base, base_log_density = base_dist(sample_size)
        sign = x_base.sum(dim=1).sign()
        neg = (sign < 0).nonzero().squeeze()
        phi_model, model_log_density = loaded_model(x_base, base_log_density, neg)

    L = int(np.sqrt(phi_model.shape[1]))

    phi_true = np.zeros((4, L, L))
    phi_true[:, training_geometry.checkerboard] = phi_model[:4, :L**2 // 2]
    phi_true[:, ~training_geometry.checkerboard] = phi_model[:4, L**2 // 2:]

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        conf = ax.imshow(phi_true[i])
        fig.colorbar(conf, ax=ax)
    
    fig.suptitle("Example configurations")

    return fig

_plot_example_configs = collect("example_configs", ("training_context",))

@figure
def plot_example_configs(_plot_example_configs):
    return _plot_example_configs[0]


@figuregen
def plot_field_components(_plot_field_components):
    yield from _plot_field_components[0]


@figure
def plot_zero_momentum_correlator(zero_momentum_correlator, training_geometry):
    """Plot zero_momentum_2pf as a function of t. Points are means across bootstrap
    sample and errorbars are standard deviations across boostrap samples
    """
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(training_geometry.length),
        y=zero_momentum_correlator.mean(axis=-1),
        yerr=zero_momentum_correlator.std(axis=-1),
        fmt="-r",
        label=f"L = {training_geometry.length}",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{G}(0, t)$")
    ax.set_xlabel("$t$")
    ax.set_title("Zero momentum two point function")
    return fig


@figure
def plot_effective_pole_mass(training_geometry, effective_pole_mass):
    """Plot effective pole mass as a function of t. The points are means
    across bootstrap samples and the errorbars are standard deviation across
    bootstrap.
    """
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(1, training_geometry.length - 1),
        y=effective_pole_mass.mean(axis=-1),
        yerr=effective_pole_mass.std(axis=-1),
        fmt="-b",
        label=f"L = {training_geometry.length}",
    )
    ax.set_ylabel("$m_p$")
    ax.set_xlabel("$t$")
    ax.set_title("Effective pole mass")
    return fig


@figure
def plot_exponential_correlation_length(
    training_geometry, exponential_correlation_length
):
    """Plot exponential correlation length as a function of t. The points are means
    across bootstrap samples and the errorbars are standard deviation across
    bootstrap.
    """
    fig, ax = plt.subplots()
    ax.errorbar(
        x=range(1, training_geometry.length - 1),
        y=exponential_correlation_length.mean(axis=-1),
        yerr=exponential_correlation_length.std(axis=-1),
        fmt="-b",
        label=f"L = {training_geometry.length}",
    )
    ax.set_ylabel("$m_p^{-2}$")
    ax.set_xlabel("$t$")
    ax.set_title("Exponential correlation length")
    return fig




@figure
def plot_two_point_correlator(two_point_correlator):
    """Represent the two point function and it's error in x and t as heatmaps
    of the respective matrices. Returns a figure with two plots, the left plot
    is the mean two point function across bootstrap and the right plot is the
    standard deviation divide by the mean (fractional error)

    """
    corr = two_point_correlator.mean(axis=-1)
    std = two_point_correlator.std(axis=-1)

    fractional_std = std / abs(corr)

    fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ax_std.set_title(r"$\sigma_G / G$")
    ax_mean.set_title("$G(x, t)$")
    ax_mean.set_xlabel("$x$")
    ax_std.set_xlabel("$x$")
    ax_mean.set_ylabel("$t$")

    im1 = ax_mean.imshow(corr)
    im2 = ax_std.imshow(fractional_std)

    ax_mean.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_mean.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_std.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_std.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.colorbar(im1, ax=ax_mean)
    fig.colorbar(im2, ax=ax_std)

    return fig


@figure
def plot_two_point_correlator_series(two_point_correlator_series, sample_interval):
    """Plot the volumn averaged two point function for the shift (0, 1)
    """
    chain_indices = np.arange(two_point_correlator_series.shape[-1]) * sample_interval
    fig, ax = plt.subplots()
    ax.set_title("Volume-averaged two point function")
    ax.set_ylabel("$G(x; t)$")
    ax.set_xlabel("$t$")
    for i in range(two_point_correlator_series.shape[0]):
        ax.plot(
            chain_indices,
            two_point_correlator_series[i],
            linestyle="-",
            linewidth=0.5,
            label=f"$x=$ (0, {i})",
        )
    ax.legend()
    return fig


@figure
def plot_two_point_correlator_autocorr(
    two_point_correlator_autocorr, two_point_correlator_optimal_window, sample_interval
):
    """Plot autocorrelation as a function of Monte Carlo time for 4 x the optimal
    window estimated by optimal_window. Mark on the optimal window as a verticle
    line
    """
    cut = max(10, 2 * np.max(two_point_correlator_optimal_window))
    chain_indices = np.arange(cut) * sample_interval

    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation of volume-averaged two point function")
    ax.set_ylabel(r"$\Gamma_G(\delta t)$")
    ax.set_xlabel("$\delta t$")

    for i in range(two_point_correlator_autocorr.shape[0]):
        color = next(ax._get_lines.prop_cycler)["color"]
        ax.plot(
            chain_indices,
            two_point_correlator_autocorr[i, :cut],
            linestyle="--",
            linewidth=0.5,
            color=color,
        )
        ax.axvline(
            two_point_correlator_optimal_window[i] * sample_interval,
            linestyle="-",
            color=color,
            label=f"$x=$ (0, {i})",
        )
    ax.set_xlim(left=0)
    ax.set_ylim(top=1)
    ax.legend()
    return fig


@figure
def plot_two_point_correlator_integrated_autocorr(
    two_point_correlator_integrated_autocorr,
    two_point_correlator_optimal_window,
    sample_interval,
):
    """plot integrated_autocorr_two_point as a function of w, up until 4 x the
    optimal window estimated by optimal_window. Mark on the optimal window as a
    verticle line

    """
    cut = max(10, 2 * np.max(two_point_correlator_optimal_window))
    chain_indices = np.arange(cut) * sample_interval
    tau = two_point_correlator_integrated_autocorr[:, two_point_correlator_optimal_window].mean()

    fig, ax = plt.subplots()
    ax.set_title("Integrated autocorrelation of volume-averaged two point function")
    ax.set_ylabel(r"$\sum \Gamma_G(\delta t)$")
    ax.set_xlabel("$\delta t$")

    for i in range(two_point_correlator_integrated_autocorr.shape[0]):
        color = next(ax._get_lines.prop_cycler)["color"]
        ax.plot(
            chain_indices,
            two_point_correlator_integrated_autocorr[i, :cut],
            linestyle="--",
            linewidth=0.5,
            color=color,
        )
        ax.axvline(
            two_point_correlator_optimal_window[i] * sample_interval,
            linestyle="-",
            color=color,
            label=f"$x=$ (0, {i})",
        )
    ax.annotate(
        fr"$\tau_G$ / {sample_interval} = {tau:.2g}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.5)
    ax.legend()
    return fig

@figure
def plot_magnetisation_series(magnetisation_series, sample_interval):
    chain_indices = np.arange(magnetisation_series.shape[-1]) * sample_interval
    fig, ax = plt.subplots()
    ax.set_title("Magnetisation")
    ax.set_ylabel("$M(t)$")
    ax.set_xlabel("$t$")
    ax.plot(
        chain_indices, magnetisation_series, linestyle="-", linewidth=0.5,
    )
    return fig

@figure
def plot_magnetisation_autocorr(
    magnetisation_autocorr, magnetisation_optimal_window, sample_interval
):
    cut = max(10, 2 * magnetisation_optimal_window)
    chain_indices = np.arange(cut) * sample_interval

    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation of magnetisation")
    ax.set_ylabel(r"$\Gamma_M(\delta t)$")
    ax.set_xlabel("$\delta t$")

    ax.plot(
        chain_indices, magnetisation_autocorr[:cut], linestyle="--", linewidth=0.5,
    )
    ax.axvline(
        magnetisation_optimal_window * sample_interval, linestyle="-", color="r",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(top=1)
    return fig


@figure
def plot_magnetisation_integrated_autocorr(
    magnetisation_integrated_autocorr,
    magnetisation_optimal_window,
    sample_interval,
):
    cut = max(10, 2 * np.max(magnetisation_optimal_window))
    chain_indices = np.arange(cut) * sample_interval
    tau = magnetisation_integrated_autocorr[magnetisation_optimal_window]

    fig, ax = plt.subplots()
    ax.set_title("Integrated autocorrelation of magnetisation")
    ax.set_ylabel(r"$\sum \Gamma_M(\delta t)$")
    ax.set_xlabel("$\delta t$")

    ax.plot(
        chain_indices,
        magnetisation_integrated_autocorr[:cut],
        linestyle="--",
        linewidth=0.5,
    )
    ax.axvline(
        magnetisation_optimal_window * sample_interval, linestyle="-", color="r",
    )
    ax.annotate(
        fr"$\tau_M$ / {sample_interval} = {tau:.2g}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.5)
    return fig

@figure
def plot_topological_charge_series(topological_charge_series, sample_interval):
    """Plot the topological charge of the ensemble as a series, ordered by the positions
    of the configurations in the Markov chain."""
    chain_indices = np.arange(topological_charge_series.shape[-1]) * sample_interval
    fig, ax = plt.subplots()
    ax.set_title("Topological charge")
    ax.set_ylabel("$Q(t)$")
    ax.set_xlabel("$t$")
    ax.plot(
        chain_indices, topological_charge_series, linestyle="-", linewidth=0.5,
    )
    return fig


@figure
def plot_topological_charge_autocorr(
    topological_charge_autocorr, topological_charge_optimal_window, sample_interval
):
    """Plot the autocorrelation of the topological charge series. Also plot a vertical
    line to denote the location of the optimal window, which minimises the error on the
    integrated autocorrelation."""
    cut = max(10, 2 * topological_charge_optimal_window)
    chain_indices = np.arange(cut) * sample_interval

    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation of topological charge")
    ax.set_ylabel(r"$\Gamma_Q(\delta t)$")
    ax.set_xlabel("$\delta t$")

    ax.plot(
        chain_indices, topological_charge_autocorr[:cut], linestyle="--", linewidth=0.5,
    )
    ax.axvline(
        topological_charge_optimal_window * sample_interval, linestyle="-", color="r",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(top=1)
    return fig


@figure
def plot_topological_charge_integrated_autocorr(
    topological_charge_integrated_autocorr,
    topological_charge_optimal_window,
    sample_interval,
):
    """Plot the integrated autocorrelation of the topological charge series. Also plot a
    vertical line to denote the location of the optimal window, which minimises the error
    on the integrated autocorrelation."""
    cut = max(10, 2 * np.max(topological_charge_optimal_window))
    chain_indices = np.arange(cut) * sample_interval
    tau = topological_charge_integrated_autocorr[topological_charge_optimal_window]

    fig, ax = plt.subplots()
    ax.set_title("Integrated autocorrelation of topological charge")
    ax.set_ylabel(r"$\sum \Gamma_Q(\delta t)$")
    ax.set_xlabel("$\delta t$")

    ax.plot(
        chain_indices,
        topological_charge_integrated_autocorr[:cut],
        linestyle="--",
        linewidth=0.5,
    )
    ax.axvline(
        topological_charge_optimal_window * sample_interval, linestyle="-", color="r",
    )
    ax.annotate(
        fr"$\tau_Q$ / {sample_interval} = {tau:.2g}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.5)
    return fig


def plot_bootstrap_single_number(observable, label):
    """Given a 1 dimensional tensor of observables, plot a histogram of the
    distribution
    """
    fig, ax = plt.subplots()
    ax.hist(
        observable,
        bins=30,
        label=f"mean: {observable.mean()}, std dev.: {observable.std()}",
    )
    title = "Bootstrap distribution: " + label
    ax.set_xlabel(label)
    ax.set_ylabel("Counts")
    ax.legend()
    ax.set_title(title)
    return fig


def plot_bootstrap_multiple_numbers(observable, labels):
    """Given a 2D tensor with boostrap sample on the final axis, yield a figure
    with a histogram for each element's bootstrap sample

    """
    for i in range(observable.shape[0]):
        yield plot_bootstrap_single_number(observable[i], labels[i])


@figure
def plot_bootstrap_two_point(two_point_correlator):
    """Plot the distribution of G(0, 0)"""
    x = t = 0
    data_to_plot = two_point_correlator(x, t)
    return plot_bootstrap_single_number(data_to_plot, rf"$G$({x},{t})")


@figure
def plot_bootstrap_susceptibility(susceptibility):
    """plot a bootstrap distribution of the susceptibility"""
    return plot_bootstrap_single_number(susceptibility, r"$\chi$")


@figure
def plot_bootstrap_ising_energy(ising_energy):
    """plot a bootstrap distribution of the ising_energy"""
    return plot_bootstrap_single_number(ising_energy, r"Ising $E$")


@figuregen
def plot_bootstrap_zero_momentum_2pf(zero_momentum_correlator):
    """For each value of t, plot a boostrap distribution of
    zero_momentum_correlator[t]

    """
    labels = [
        r"$\tilde{G}$" + f"$(0,{t})$" for t in range(zero_momentum_correlator.shape[0])
    ]
    yield from plot_bootstrap_multiple_numbers(zero_momentum_correlator, labels)


@figuregen
def plot_bootstrap_effective_pole_mass(effective_pole_mass):
    """For each value of t from 1 to L-2, plot a bootstrap distribution of
    effective_pole_mass[t]

    """
    labels = [r"$m_p^{eff}$" + f"$({t})$" for t in range(effective_pole_mass.shape[0])]
    yield from plot_bootstrap_multiple_numbers(effective_pole_mass, labels)
