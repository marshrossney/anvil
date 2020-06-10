"""
models.py

Module containing reportengine actions which return callable objects that execute
normalising flows constructed from multiple layers via function composition.
"""
from functools import partial

from anvil.core import Sequential

import anvil.layers as layers

from math import pi


def support(target_dist):
    """Return the support of the target distribution."""
    # NOTE: may need to rethink this for multivariate distributions
    return target_dist.support


def coupling_pair(coupling_layer, size_half, **layer_spec):
    """Helper function which returns a callable object that performs a coupling
    transformation on both even and odd lattice sites."""
    coupling_transformation = partial(coupling_layer, size_half, **layer_spec)
    return Sequential(
        coupling_transformation(even_sites=True),
        coupling_transformation(even_sites=False),
    )


def real_nvp(
    size_half,
    n_affine,
    hidden_shape=[24,],
    activation="leaky_relu",
    s_final_activation="leaky_relu",
    batch_normalise=False,
):
    """Action that returns a callable object that performs a sequence of `n_affine`
    affine coupling transformations on both partitions of the input vector."""
    affine_pairs = [
        coupling_pair(
            layers.AffineLayer,
            size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            s_final_activation=s_final_activation,
            batch_normalise=batch_normalise,
        )
        for _ in range(n_affine)
    ]
    return Sequential(*affine_pairs)


def real_nvp_circle(size_half, real_nvp):
    """Action that returns a callable object that projects an input vector from 
    (0, 2\pi)->R1, performs a sequence of affine transformations, then does the
    inverse projection back to (0, 2\pi)"""
    return Sequential(
        layers.ProjectionLayer(), real_nvp, layers.InverseProjectionLayer()
    )


def real_nvp_sphere(size_half, real_nvp):
    """Action that returns a callable object that projects an input vector from 
    S2 - {0} -> R2, performs a sequence of affine transformations, then does the
    inverse projection back to S2 - {0}"""
    return Sequential(
        layers.ProjectionLayer2D(size_half),
        real_nvp,
        layers.InverseProjectionLayer2D(size_half),
    )


def linear_spline(
    size_half,
    support=(0, 1),
    n_segments=4,
    hidden_shape=[24,],
    activation="leaky_relu",
    batch_normalise=False,
):
    """Action that returns a callable object that performs a pair of linear spline
    transformations, one on each half of the input vector."""
    return Sequential(
        coupling_pair(
            layers.LinearSplineLayer,
            size_half,
            n_segments=n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
            batch_normalise=batch_normalise,
        ),
        layers.GlobalAffineLayer(scale=support[1] - support[0], shift=support[0]),
    )


def rational_quadratic_spline(
    size_half,
    support=(1, 1),
    n_segments=4,
    hidden_shape=[24,],
    activation="leaky_relu",
    batch_normalise=False,
):
    """Action that returns a callable object that performs a pair of linear spline
    transformations, one on each half of the input vector."""
    return coupling_pair(
            layers.CircularSplineLayer,
            size_half,
            n_segments=n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
            batch_normalise=batch_normalise,
        )


MODEL_OPTIONS = {
    "real_nvp": real_nvp,
    "real_nvp_circle": real_nvp_circle,
    "real_nvp_sphere": real_nvp_sphere,
    "linear_spline": linear_spline,
    "rational_quadratic_spline": rational_quadratic_spline,
}
