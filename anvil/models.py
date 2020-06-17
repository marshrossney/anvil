"""
models.py

Module containing reportengine actions which return callable objects that execute
normalising flows constructed from multiple layers via function composition.
"""
from anvil.core import Sequential, RBSequential, RBLayerNd, AutoregressiveLayer
import anvil.layers as layers


def affine_layer_spec(
    hidden_shape=[24,],
    activation="leaky_relu",
    s_final_activation="leaky_relu",
    batch_normalise=False,
):
    return dict(
        hidden_shape=hidden_shape,
        activation=activation,
        s_final_activation=s_final_activation,
        batch_normalise=batch_normalise,
    )


def ncp_layer_spec(
    hidden_shape=[24,], activation="leaky_relu", batch_normalise=False,
):
    return dict(
        hidden_shape=hidden_shape,
        activation=activation,
        batch_normalise=batch_normalise,
    )


def spline_layer_spec(
    n_segments=4, hidden_shape=[24,], activation="leaky_relu", batch_normalise=False,
):
    return dict(
        n_segments=n_segments,
        hidden_shape=hidden_shape,
        activation=activation,
        batch_normalise=batch_normalise,
    )


def real_nvp(n_lattice, affine_layer_spec, n_affine=2):
    """Action that returns a callable object that performs a sequence of `n_affine`
    affine coupling transformations on both partitions of the input vector."""
    data_shape = (1, n_lattice // 2)
    return RBSequential(
        *[
            layers.AffineLayer(data_shape, data_shape, **affine_layer_spec)
            for _ in range(n_affine)
        ]
    )


def real_nvp_circle(real_nvp):
    """Action that returns a callable object that projects an input vector from 
    (0, 2\pi)->R1, performs a sequence of affine transformations, then does the
    inverse projection back to (0, 2\pi)"""
    return Sequential(
        layers.ProjectionLayer(), real_nvp, layers.InverseProjectionLayer()
    )


def real_nvp_sphere_old(n_lattice, affine_layer_spec, n_affine=2):
    """Action that returns a callable object that projects an input vector from 
    S2 - {0} -> R2, performs a sequence of affine transformations, then does the
    inverse projection back to S2 - {0}"""
    data_shape = (2, n_lattice // 2)
    return Sequential(
        layers.ProjectionLayer2D(),
        RBSequential(
            *[
                layers.AffineLayer(data_shape, data_shape, **affine_layer_spec)
                for _ in range(n_affine)
            ]
        ),
        layers.InverseProjectionLayer2D(),
    )


def ncp_circle(
    n_lattice,
    ncp_layer_spec,
    n_couple=1,  # unlikely that function composition is beneficial
):
    """Action that returns a callable object that performs a sequence of transformations
    from (0, 2\pi) -> (0, 2\pi), each of which are the composition of a stereographic
    projection transformation, an affine transformation, and the inverse projection."""
    data_shape = (1, n_lattice // 2)
    return RBSequential(
        *[
            layers.NCPLayer(data_shape, data_shape, **ncp_layer_spec)
            for _ in range(n_couple)
        ]
    )


def linear_spline(
    n_lattice, spline_layer_spec,
):
    """Action that returns a callable object that performs a pair of linear spline
    transformations, one on each half of the input vector."""
    data_shape = (1, n_lattice // 2)
    return RBSequential(
        layers.LinearSplineLayer(data_shape, data_shape, **spline_layer_spec),
        layers.LinearSplineLayer(data_shape, data_shape, **spline_layer_spec),
    )


def quadratic_spline(
    n_lattice, spline_layer_spec,
):
    """Action that returns a callable object that performs a pair of quadratic spline
    transformations, one on each half of the input vector."""
    data_shape = (1, n_lattice // 2)
    return RBSequential(
        layers.QuadraticSplineLayer(data_shape, data_shape, **spline_layer_spec),
        layers.QuadraticSplineLayer(data_shape, data_shape, **spline_layer_spec),
    )


def circular_spline(n_lattice, spline_layer_spec):
    """Action that returns a callable object that performs a pair of quadratic spline
    transformations, one on each half of the input vector."""
    data_shape = (1, n_lattice // 2)
    return RBSequential(
        layers.CircularSplineLayer(data_shape, data_shape, **spline_layer_spec),
        layers.CircularSplineLayer(data_shape, data_shape, **spline_layer_spec),
    )


def spherical_spline_old(n_lattice, spline_layer_spec):
    return RBSequential(
        RBLayerNd(
            [layers.QuadraticSplineLayer, layers.CircularSplineLayer],
            n_lattice,
            spline_layer_spec,
        ),
        RBLayerNd(
            [layers.QuadraticSplineLayer, layers.CircularSplineLayer],
            n_lattice,
            spline_layer_spec,
        ),
    )


def spherical_spline(n_lattice, spline_layer_spec):
    return AutoregressiveLayer(
        [layers.QuadraticSplineLayer, layers.CircularSplineLayer],
        n_lattice,
        spline_layer_spec,
    )


def real_nvp_sphere(n_lattice, affine_layer_spec, n_affine=2):
    """Action that returns a callable object that projects an input vector from 
    S2 - {0} -> R2, performs a sequence of affine transformations, then does the
    inverse projection back to S2 - {0}"""
    return Sequential(
        layers.ProjectionLayer2D(),
        AutoregressiveLayer(
            [layers.AffineLayer, layers.AffineLayer],
            n_lattice,
            affine_layer_spec,
            i_start=0,
            n_redblack=n_affine,
        ),
        AutoregressiveLayer(
            [layers.AffineLayer, layers.AffineLayer],
            n_lattice,
            affine_layer_spec,
            i_start=1,
            n_redblack=n_affine,
        ),
        layers.InverseProjectionLayer2D(),
    )


MODEL_OPTIONS = {
    "real_nvp": real_nvp,
    "real_nvp_circle": real_nvp_circle,
    "real_nvp_sphere": real_nvp_sphere,
    "ncp_circle": ncp_circle,
    "linear_spline": linear_spline,
    "quadratic_spline": quadratic_spline,
    "circular_spline": circular_spline,
    "spherical_spline": spherical_spline,
}
