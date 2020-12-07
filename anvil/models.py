"""
models.py

Module containing reportengine actions which return callable objects that execute
normalising flows constructed from multiple layers via function composition.
"""
from functools import partial

from anvil.core import Sequential

import anvil.layers as layers


def target_support(target_dist):
    """Return the support of the target distribution."""
    # NOTE: may need to rethink this for multivariate distributions
    return target_dist.support


def coupling_pair(coupling_layer, i, size_half, **layer_spec):
    """Helper function which returns a callable object that performs a coupling
    transformation on both even and odd lattice sites."""
    coupling_transformation = partial(coupling_layer, i, size_half, **layer_spec)
    return Sequential(
        coupling_transformation(even_sites=True),
        coupling_transformation(even_sites=False),
    )


def real_nvp(
    size_half,
    n_affine,
    hidden_shape=[
        24,
    ],
    activation="tanh",
    s_final_activation=None,
    symmetric_networks=True,
    bnorm=False,
    bn_scale=1.0,
):
    """Action that returns a callable object that performs a sequence of `n_affine`
    affine coupling transformations on both partitions of the input vector."""
    if bnorm == False:
        affine_pairs = [
            coupling_pair(
                layers.AffineLayer,
                i + 1,
                size_half,
                hidden_shape=hidden_shape,
                activation=activation,
                s_final_activation=s_final_activation,
                symmetric_networks=symmetric_networks,
            )
            for i in range(n_affine)
        ]
        return Sequential(*affine_pairs)
    else:
        output = []
        for i in range(n_affine):
            output.append(
                coupling_pair(
                    layers.AffineLayer,
                    i + 1,
                    size_half,
                    hidden_shape=hidden_shape,
                    activation=activation,
                    s_final_activation=s_final_activation,
                    symmetric_networks=symmetric_networks,
                )
            )
            if i < n_affine - 1:
                output.append(layers.BatchNormLayer(bn_scale, learnable=False))
        return Sequential(*output)


def real_nvp_shift(real_nvp, shift_init=0.0):
    return Sequential(
        real_nvp, layers.GlobalAdditiveLayer(shift_init=shift_init, learnable=True)
    )


def nice(
    size_half,
    n_additive,
    hidden_shape=[
        24,
    ],
    activation="tanh",
    symmetric=True,
    bnorm=False,
    bn_scale=1.0,
):
    """Action that returns a callable object that performs a sequence of `n_affine`
    affine coupling transformations on both partitions of the input vector."""
    if bnorm == False:
        pairs = [
            coupling_pair(
                layers.AdditiveLayer,
                i + 1,
                size_half,
                hidden_shape=hidden_shape,
                activation=activation,
                symmetric=symmetric,
            )
            for i in range(n_additive)
        ]
        return Sequential(*pairs)
    else:
        output = []
        for i in range(n_additive):
            output.append(
                coupling_pair(
                    layers.AdditiveLayer,
                    i + 1,
                    size_half,
                    hidden_shape=hidden_shape,
                    activation=activation,
                    symmetric=symmetric,
                )
            )
            if i < n_additive - 1:
                output.append(layers.BatchNormLayer(bn_scale, learnable=True))
        return Sequential(*output)


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


def ncp_circle(
    size_half,
    n_layers=1,  # unlikely that function composition is beneficial
    hidden_shape=[
        24,
    ],
    activation="leaky_relu",
    s_final_activation=None,
):
    """Action that returns a callable object that performs a sequence of transformations
    from (0, 2\pi) -> (0, 2\pi), each of which are the composition of a stereographic
    projection transformation, an affine transformation, and the inverse projection."""
    ncp_pairs = [
        coupling_pair(
            layers.NCPLayer,
            size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            s_final_activation=s_final_activation,
        )
        for _ in range(n_layers)
    ]
    return Sequential(*ncp_pairs)


def linear_spline(
    size_half,
    target_support,
    n_segments=4,
    hidden_shape=[
        24,
    ],
    activation="leaky_relu",
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
        ),
        layers.GlobalAffineLayer(
            scale=target_support[1] - target_support[0], shift=target_support[0]
        ),
    )


def quadratic_spline(
    size_half,
    target_support,
    n_segments=4,
    hidden_shape=[
        24,
    ],
    activation="leaky_relu",
):
    """Action that returns a callable object that performs a pair of linear spline
    transformations, one on each half of the input vector."""
    return Sequential(
        coupling_pair(
            layers.QuadraticSplineLayer,
            size_half,
            n_segments=n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
        ),
        layers.GlobalAffineLayer(
            scale=target_support[1] - target_support[0], shift=target_support[0]
        ),
    )


def rational_quadratic_spline(
    size_half,
    n_pairs=1,
    interval=2,
    n_segments=4,
    hidden_shape=[
        24,
    ],
    activation="tanh",
    symmetric_spline=False,
):
    """Action that returns a callable object that performs a pair of circular spline
    transformations, one on each half of the input vector."""
    return Sequential(
        # layers.GlobalAffineLayer(-1, 0),
        *[
            coupling_pair(
                layers.RationalQuadraticSplineLayer,
                i + 3,
                size_half,
                interval=interval,
                n_segments=n_segments,
                hidden_shape=hidden_shape,
                activation=activation,
                symmetric_spline=symmetric_spline,
            )
            for i in range(n_pairs)
        ]
    )


def rqs_shift(rational_quadratic_spline, shift_init=0.0):
    return Sequential(
        rational_quadratic_spline,
        layers.GlobalAdditiveLayer(shift_init=shift_init, learnable=True),
    )


def circular_spline(
    size_half,
    n_pairs,
    n_segments=4,
    hidden_shape=[24],
    activation="leaky_relu",
):
    """Action that returns a callable object that performs a pair of circular spline
    transformations, one on each half of the input vector."""
    return Sequential(
        *[
            coupling_pair(
                layers.CircularSplineLayer,
                i,
                size_half,
                n_segments=n_segments,
                hidden_shape=hidden_shape,
                activation=activation,
            )
            for i in range(n_pairs)
        ]
    )


def spline_affine(real_nvp, rational_quadratic_spline):
    return Sequential(rational_quadratic_spline, real_nvp)


def affine_spline(
    real_nvp, rational_quadratic_spline, sigma, scale_sigma_before_spline=1
):
    return Sequential(
        real_nvp,
        layers.BatchNormLayer(scale=scale_sigma_before_spline * sigma),
        rational_quadratic_spline,
    )


def spline_sandwich(
    rational_quadratic_spline,
    sigma,
    size_half,
    n_affine,
    hidden_shape=[
        24,
    ],
    activation="tanh",
    s_final_activation=None,
    symmetric_networks=True,
    scale_sigma_before_spline=1,
):
    affine_1 = [
        coupling_pair(
            layers.AffineLayer,
            i + 1,
            size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            s_final_activation=s_final_activation,
            symmetric_networks=symmetric_networks,
        )
        for i in range(n_affine)
    ]
    affine_2 = [
        coupling_pair(
            layers.AffineLayer,
            i + 3,
            size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            s_final_activation=s_final_activation,
            symmetric_networks=symmetric_networks,
        )
        for i in range(n_affine)
    ]
    return Sequential(
        *affine_1,
        layers.BatchNormLayer(scale=scale_sigma_before_spline * sigma),
        rational_quadratic_spline,
        *affine_2,
    )


def sandwich_shift(spline_sandwich, shift_init=0.0):
    return Sequential(
        spline_sandwich,
        layers.GlobalAdditiveLayer(shift_init=shift_init, learnable=True),
    )


MODEL_OPTIONS = {
    "real_nvp": real_nvp,
    "real_nvp_circle": real_nvp_circle,
    "real_nvp_sphere": real_nvp_sphere,
    "linear_spline": linear_spline,
    "quadratic_spline": quadratic_spline,
    "rational_quadratic_spline": rational_quadratic_spline,
    "circular_spline": circular_spline,
    "ncp_circle": ncp_circle,
    "spline_affine": spline_affine,
    "affine_spline": affine_spline,
    "spline_sandwich": spline_sandwich,
    "nice": nice,
    "real_nvp_shift": real_nvp_shift,
    "rqs_shift": rqs_shift,
    "sandwich_shift": sandwich_shift,
}
