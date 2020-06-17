# -*- coding: utf-8 -*-
r"""
layers.py

Contains nn.Modules which implement transformations of an input vector whilst computing the
Jacobian determinant of the inverse transformation.

Each transformation layers may contain several neural networks or learnable parameters.

A normalising flow, f, can be constructed from multiple layers using function composition:

        f(\phi) = g_1( g_2( ... ( g_n( \phi ) ) ... ) )

which is implemented using the architecture provided by torch.nn

Layers can be divided into two classes.

1. Coupling layers: (x_in, x_passive, log_density) -> (phi_out, log_density)

    Takes two tensors `x_in`, `x_passive` of dimensions (n_batch, D) and one tensor `log_density`
    of dimensions (n_batch, 1).

    The `x_in` is transformed by a bijective function whose parameters are neural networks
    which themselves take `x_passive` as an input vector.

    The current value of the logarithm of the probability density is updated by adding the
    Jacobian determinant of the *inverse* of the coupling transformation.

2. Full layers: (x_in, log_density) -> (phi_out, log_density)

    Takes a single input tensor and transforms the entire tensor, with no coupling occuring
    between lattice sites.

    These layers may be 'cascaded' using anvil.core.Sequential provided `x_in` is the full
    dataset.
"""
import torch
import torch.nn as nn
from torchsearchsorted import searchsorted
from math import pi

from anvil.core import NeuralNetwork
from anvil.utils import prod

# ----------------------------------------------------------------------------------------- #
#                                                                                           #
#                                   Coupling layers                                         #
#                                                                                           #
# ----------------------------------------------------------------------------------------- #


class AffineLayer(nn.Module):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    An affine transformation, x = g_i(\phi), is defined as:

        x_r = \phi_r
        x_in = \phi_b * exp(s_i(\phi_r)) + t_i(\phi_r)

    Parameters
    ----------
    shape_in: int
        Size of the passive partition at dimension 1, which is also the size of the input
        vector for the neural networks.
    shape_out: int
        Size of the active partition, being transformed by the spline layer, at dimension 1,
        which is also the number of rows in the matrix output by the neural networks.
    hidden_shape: list
        list containing hidden vector sizes for the neural networks.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the networks.
    s_final_activation: str
        string which is a key for an activation function, which the output of the s
        network will be passed through.
    batch_normalise: bool
        flag indicating whether or not to use batch normalising within the neural
        networks.

    Attributes
    ----------
    s_network: torch.nn.Module
        the dense layers of network s, values are intialised as per the default
        initialisation of `nn.Linear`
    t_network: torch.nn.Module
        the dense layers of network t, values are intialised as per the default
        initialisation of `nn.Linear`

    Methods
    -------
    forward(x_in, x_passive, log_density)
        Performs an affine transformation of the `x_in` tensor, using the `x_passive`
        tensor as parameters for the neural networks. Returns the transformed tensor
        along with the updated log density.
    """

    def __init__(
        self,
        shape_in: int,
        shape_passive: tuple,
        *,
        hidden_shape: list,
        activation: str,
        s_final_activation: str,
        batch_normalise: bool,
    ):
        super().__init__()

        self.s_network = NeuralNetwork(
            shape_in=shape_passive,
            shape_out=shape_in,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=s_final_activation,
            batch_normalise=batch_normalise,
        )
        self.t_network = NeuralNetwork(
            shape_in=shape_passive,
            shape_out=shape_in,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            batch_normalise=batch_normalise,
        )

    def forward(self, x_in, x_passive, log_density):
        """Forward pass of affine transformation."""
        s_out = self.s_network(x_passive)
        t_out = self.t_network(x_passive)

        phi_out = (x_in - t_out) * torch.exp(-s_out)
        log_density += s_out.sum(dim=2).sum(dim=1, keepdim=True)

        return phi_out, x_passive, log_density


class NCPLayer(nn.Module):
    r"""A coupling layer which performs a transformation from (0, 2\pi) -> (0, 2\pi),
    which is an affine transformation X -> \alpha * X + \beta on the Euclidean vector
    obtained by stereographic projection of the input data.

    The transformation x = g(\phi) is defined as

        x_r = \phi_r
        x_in = (
                2 \arctan( \alpha(\phi_r) + \tan((\phi_b - \pi) / 2) + \beta(\phi_r))
                + \pi + \theta
              )

    where \alpha and \beta are neural networks, and \theta is a learnable global phase shift.

    The Jacobian determinant can be computed analytically for the full transformation,
    including projection and inverse. 
        
        | \det J | = \prod_n (
            (1 + \beta ** 2) / \alpha * \sin^2(x / 2) + \beta 
            + \alpha * \cos^2(x / 2)
            - \beta * \sin(x_in)
            )
    
    Parameters
    ----------
    shape_in: int
        Size of the passive partition at dimension 1, which is also the size of the input
        vector for the neural networks.
    shape_out: int
        Size of the active partition, being transformed by the spline layer, at dimension 1,
        which is also the number of rows in the matrix output by the neural networks.
    hidden_shape: list
        list containing hidden vector sizes for the neural networks.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the networks.
    batch_normalise: bool
        flag indicating whether or not to use batch normalising within the neural
        networks.

    Attributes
    ----------
    s_network: NeuralNetwork
        When exponentiated, this is the linear transformation \alpha.
    t_network: NeuralNetwork
        The shift part (\beta) of the affine transformation.
    phase_shift: nn.Parameter
        A learnable global phase shift.
    
    Methods
    -------
    forward(x_in, x_passive, log_density)
        Performs an affine transformation of the `x_in` tensor, using the `x_passive`
        tensor as parameters for the neural networks. Returns the transformed tensor
        along with the updated log density.

    Notes
    -----
    This can be thought of as a lightweight version of real_nvp_circle, with just one affine
    transformation.
    """

    def __init__(
        self,
        shape_in: tuple,
        shape_passive: int,
        *,
        hidden_shape: list,
        activation: str,
        batch_normalise: bool,
    ):
        super().__init__()

        self.s_network = NeuralNetwork(
            shape_in=shape_passive,
            shape_out=shape_in,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            batch_normalise=batch_normalise,
        )
        self.t_network = NeuralNetwork(
            shape_in=shape_passive,
            shape_out=shape_in,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            batch_normalise=batch_normalise,
        )
        self.phase_shift = nn.Parameter(torch.rand(1))

    def forward(self, x_in, x_passive, log_density):
        """Forward pass of the project-affine-inverse transformation."""
        alpha = torch.exp(self.s_network(x_passive))
        beta = self.t_network(x_passive)

        phi_out = (
            2 * torch.atan(alpha * torch.tan((x_in - pi) / 2) + beta)
            + pi
            + self.phase_shift
        ) % (2 * pi)

        log_density += (
            torch.log(
                (1 + beta ** 2) / alpha * torch.sin(x_in / 2) ** 2
                + alpha * torch.cos(x_in / 2) ** 2
                - beta * torch.sin(x_in)
            )
            .sum(dim=2)
            .sum(dim=1, keepdim=True)
        )

        return phi_out, x_passive, log_density


class LinearSplineLayer(nn.Module):
    r"""A coupling transformation from [0, 1] -> [0, 1] based on a piecewise linear function.

    The interval is divided into K equal-width (w) segments (bins), with K+1 knot points
    (bin boundaries). The coupling transformation is defined piecewise by the unique
    polynomials whose end-points are the knot points.

    A neural network generates the K values for the y-positions (heights) at the knot points.
    The coupling transformation is then defined as the cumulative distribution function
    associated with the probability masses given by the heights.

    The inverse coupling transformation is

        \phi = C^{-1}(x, {p_k}) = \phi_{k-1} + \alpha p_k

    where \phi_{k-1} = \sum_{k'=1}^{k-1} p_{k'} is the (k-1)-th knot point, and \alpha is the
    fractional position of x in the k-th bin, which is (x - (k-1) * w) / w.

    The gradient of the forward transformation is simply
        
        dx / d\phi = w / p_k

    Parameters
    ----------
    shape_in: int
        Size of the passive partition at dimension 1, which is also the size of the input
        vector for the neural networks.
    shape_out: int
        Size of the active partition, being transformed by the spline layer, at dimension 1,
        which is also the number of rows in the matrix output by the neural networks.
    n_segments: int
        Number of segments (bins).
    hidden_shape: list
        list containing hidden vector sizes the neural network.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the network.
    batch_normalise: bool
        flag indicating whether or not to use batch normalising within the neural
        network.

    Attributes
    ----------
    network: torch.nn.Module
        the dense layers of network h, values are intialised as per the default
        initialisation of `nn.Linear`
    
    Methods
    -------
    forward(x_in, x_passive, log_density)
        Performs an affine transformation of the `x_in` tensor, using the `x_passive`
        tensor as parameters for the neural networks. Returns the transformed tensor
        along with the updated log density.
    """

    def __init__(
        self,
        shape_in: tuple,
        shape_passive: tuple,
        *,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        batch_normalise: bool,
    ):
        super().__init__()
        self.shape_in = shape_in
        self.size_in = prod(shape_in)

        self.n_segments = n_segments
        self.width = 1 / n_segments

        eps = 1e-6  # prevent rounding error which causes sorting into -1th bin
        self.x_knot_points = torch.linspace(-eps, 1 + eps, n_segments + 1).view(1, -1)

        self.network = NeuralNetwork(
            shape_in=shape_passive,
            shape_out=(self.size_in, n_segments),
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=activation,
            batch_normalise=batch_normalise,
        )
        self.norm_func = nn.Softmax(dim=2)

        self.scale = pi  # TODO: improve

    def forward(self, x_in, x_passive, log_density):
        """Forward pass of the linear spline layer."""
        # Mix component and lattice dimensions so that searchsorted can work
        x_in = x_in.view(-1, self.size_in) / self.scale

        net_out = self.norm_func(self.network(x_passive))
        phi_knot_points = torch.cat(
            (torch.zeros_like(x_in).unsqueeze(dim=2), torch.cumsum(net_out, dim=2)),
            dim=2,
        )

        # Sort x_in into the appropriate bin
        # NOTE: x_in must be contiguous, otherwise searchsorted returns nonsense
        k_ind = (
            (searchsorted(self.x_knot_points, x_in.contiguous()) - 1)
            .clamp(
                0, self.n_segments - 1
            )  # TODO: would be nice to lose this extra safety net
            .view(-1, self.size_in, 1)
        )
        x_in.unsqueeze_(dim=2)

        p_k = torch.gather(net_out, 2, k_ind)
        alpha = (x_in - k_ind * self.width) / self.width
        phi_km1 = torch.gather(phi_knot_points, 2, k_ind)

        # Un-mix component and lattice dimensions
        phi_out = phi_km1 + alpha * p_k
        log_density -= torch.log(p_k).sum(dim=1)

        phi_out = phi_out.view(-1, *self.shape_in) * self.scale

        return phi_out, x_passive, log_density


class QuadraticSplineLayer(nn.Module):
    r"""A coupling transformation from [0, 1] -> [0, 1] based on a piecewise quadratic function.

    The interval is divided into K segments (bins), with K+1 knot points (bin boundaries).
    The coupling transformation is defined piecewise by the unique polynomials whose
    end-points are the knot points.

    A neural network generates K+1 values for the y-positions (heights) at the x knot points,
    and K bin widths. The inverse coupling transformation is then defined as the cumulative
    distribution function associated with the piecewise linear probability density function
    obtained by interpolating between the heights.

    The result is a quadratic function defined piecewise by 

        \phi = C^{-1}(x, {h_k}) = \phi_{k-1} + \alpha(x) h_k w_k
                                  + \alpha(x)^2 / 2 * (h_{k+1} - h_k) w_k

    where \alpha(x) is the fractional position within a bin,

        \alpha(x) = (x - x_{k-1}) / w_k

    Parameters
    ----------
    shape_in: int
        Size of the passive partition at dimension 1, which is also the size of the input
        vector for the neural networks.
    shape_out: int
        Size of the active partition, being transformed by the spline layer, at dimension 1,
        which is also the number of rows in the matrix output by the neural networks.
    n_segments: int
        Number of segments (bins).
    hidden_shape: list
        list containing hidden vector sizes the neural network.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the network.
    batch_normalise: bool
        flag indicating whether or not to use batch normalising within the neural
        network.

    Attributes
    ----------
    network: torch.nn.Module
        the dense layers of network h, values are intialised as per the default
        initialisation of `nn.Linear`

    Methods
    -------
    forward(x_in, x_passive, log_density)
        Performs an affine transformation of the `x_in` tensor, using the `x_passive`
        tensor as parameters for the neural networks. Returns the transformed tensor
        along with the updated log density.
    """

    def __init__(
        self,
        shape_in: int,
        shape_passive: int,
        *,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        batch_normalise: bool,
    ):
        super().__init__()
        self.shape_in = shape_in
        self.size_in = prod(shape_in)

        self.n_segments = n_segments

        self.network = NeuralNetwork(
            shape_in=shape_passive,
            shape_out=(self.size_in, 2 * n_segments + 1),
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=activation,
            batch_normalise=batch_normalise,
        )
        self.w_norm_func = nn.Softmax(dim=2)

        self.eps = 1e-6  # prevent rounding error which causes sorting into -1th bin

        self.scale = pi  # TODO: improve

    @staticmethod
    def h_norm_func(h_raw, w_norm):
        """Normalisation function for height values."""
        return torch.exp(h_raw) / (
            0.5 * w_norm * (torch.exp(h_raw[..., :-1]) + torch.exp(h_raw[..., 1:]))
        ).sum(dim=2, keepdim=True)

    def forward(self, x_in, x_passive, log_density):
        """Forward pass of the quadratic spline layer."""
        x_in = x_in.view(-1, self.size_in) / self.scale

        h_raw, w_raw = self.network(x_passive).split(
            (self.n_segments + 1, self.n_segments), dim=2
        )
        w_norm = self.w_norm_func(w_raw)
        h_norm = self.h_norm_func(h_raw, w_norm)

        x_knot_points = torch.cat(
            (
                torch.zeros_like(x_in).unsqueeze(dim=2) - self.eps,
                torch.cumsum(w_norm, dim=2),
            ),
            dim=2,
        )
        phi_knot_points = torch.cat(
            (
                torch.zeros_like(x_in).unsqueeze(dim=2),
                torch.cumsum(
                    0.5 * w_norm * (h_norm[..., :-1] + h_norm[..., 1:]), dim=2,
                ),
            ),
            dim=2,
        )

        # Temporarily mix batch, component and lattice dimensions so that the
        # bisection search can be done in a single operation
        k_ind = (
            (
                searchsorted(
                    x_knot_points.contiguous().view(-1, self.n_segments + 1),
                    x_in.contiguous().view(-1, 1),
                )
                - 1
            )
            .clamp(0, self.n_segments - 1)
            .view(-1, self.size_in, 1)
        )
        x_in.unsqueeze_(dim=2)

        w_k = torch.gather(w_norm, 2, k_ind)
        h_k = torch.gather(h_norm, 2, k_ind)
        h_kp1 = torch.gather(h_norm, 2, k_ind + 1)

        x_km1 = torch.gather(x_knot_points, 2, k_ind)
        phi_km1 = torch.gather(phi_knot_points, 2, k_ind)
        alpha = (x_in - x_km1) / w_k

        phi_out = phi_km1 + alpha * h_k * w_k + 0.5 * alpha.pow(2) * (h_kp1 - h_k) * w_k
        log_density -= torch.log(h_k + alpha * (h_kp1 - h_k)).sum(dim=1)

        phi_out = phi_out.view(-1, *self.shape_in) * self.scale

        return phi_out, x_passive, log_density


class CircularSplineLayer(nn.Module):
    r"""A coupling transformation from S^1 -> S^1 based on a piecewise rational quadratic
    spline function.

    The interval is divided into K segments (bins) with widths w_k and heights h_k. The
    'knot points' (\phi_k, x_k) are the cumulative sum of (h_k, w_k), starting at (0, 0)
    and ending at (2\pi, 2\pi).

    In addition to the w_k and h_k, the derivatives d_k at the knot points are generated
    by a neural network. d_0 is set equal to d_K.

    Defing the slopes s_k = h_k / w_k and fractional position within a bin
        
            alpha(x) = (x - x_{k-1}) / w_k

    the coupling transformation is defined piecewise by

            \phi = C^{-1}(x, {h_k, s_k, d_k})
                 = \phi_{k-1}
                 + ( h_k(s_k * \alpha^2 + d_k * \alpha * (1 - \alpha)) )
                 / ( s_k + (d_{k+1} + d_k - 2s_k) * \alpha * (1 - \alpha) )

    Parameters
    ----------
    shape_in: int
        Size of the passive partition at dimension 1, which is also the size of the input
        vector for the neural networks.
    shape_out: int
        Size of the active partition, being transformed by the spline layer, at dimension 1,
        which is also the number of rows in the matrix output by the neural networks.
    n_segments: int
        Number of segments (bins).
    hidden_shape: list
        list containing hidden vector sizes the neural network.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the network.
    batch_normalise: bool
        flag indicating whether or not to use batch normalising within the neural
        network.

    Attributes
    ----------
    network: torch.nn.Module
        the dense layers of the network, values are intialised as per the default
        initialisation of `nn.Linear`
    phase_shift: torch.nn.Parameter
        a learnable phase shift which incurs no Jacobian penalty, the purpose being 
        that 0 and 2\pi are no longer fixed points of the transformation.

    Methods
    -------
    forward(x_in, x_passive, log_density)
        Performs an affine transformation of the `x_in` tensor, using the `x_passive`
        tensor as parameters for the neural networks. Returns the transformed tensor
        along with the updated log density.
    """

    def __init__(
        self,
        shape_in: tuple,
        shape_passive: tuple,
        *,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        batch_normalise: bool,
    ):
        super().__init__()
        self.shape_in = shape_in
        self.size_in = prod(shape_in)

        self.n_segments = n_segments

        self.network = NeuralNetwork(
            shape_in=shape_passive,
            shape_out=(self.size_in, 3 * n_segments),
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=activation,
            batch_normalise=batch_normalise,
        )
        self.phase_shift = nn.Parameter(torch.rand(1))

        self.norm_func = nn.Softmax(dim=2)
        self.softplus = nn.Softplus()

        self.eps = 1e-6

    def forward(self, x_in, x_passive, log_density):
        """Forward pass of the rational quadratic spline layer."""
        x_in = x_in.view(-1, self.size_in)

        h_raw, w_raw, d_raw = self.network(x_passive).split(
            (self.n_segments, self.n_segments, self.n_segments), dim=2
        )
        h_norm = self.norm_func(h_raw) * 2 * pi
        w_norm = self.norm_func(w_raw) * 2 * pi
        d_norm = self.softplus(d_raw)

        x_knot_points = torch.cat(
            (
                torch.zeros_like(x_in).unsqueeze(dim=2) - self.eps,
                torch.cumsum(w_norm, dim=2),
            ),
            dim=2,
        )
        phi_knot_points = torch.cat(
            (torch.zeros_like(x_in).unsqueeze(dim=2), torch.cumsum(h_norm, dim=2),),
            dim=2,
        )

        k_ind = (
            (
                searchsorted(
                    x_knot_points.contiguous().view(-1, self.n_segments + 1),
                    x_in.contiguous().view(-1, 1),
                )
                - 1
            )
            .clamp(0, self.n_segments - 1)
            .view(-1, self.size_in, 1)
        )
        x_in.unsqueeze_(dim=2)

        w_k = torch.gather(w_norm, 2, k_ind)
        h_k = torch.gather(h_norm, 2, k_ind)
        s_k = h_k / w_k
        d_k = torch.gather(d_norm, 2, k_ind)
        d_kp1 = torch.gather(d_norm, 2, (k_ind + 1) % self.n_segments)

        x_km1 = torch.gather(x_knot_points, 2, k_ind)
        phi_km1 = torch.gather(phi_knot_points, 2, k_ind)

        alpha = (x_in - x_km1) / w_k

        phi_out = (
            phi_km1
            + (h_k * (s_k * alpha.pow(2) + d_k * alpha * (1 - alpha)))
            / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha))
        ).view(-1, *self.shape_in)
        phi_out = (phi_out + self.phase_shift) % (2 * pi)

        grad = (
            s_k.pow(2)
            * (
                d_kp1 * alpha.pow(2)
                + 2 * s_k * alpha * (1 - alpha)
                + d_k * (1 - alpha).pow(2)
            )
        ) / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha)).pow(2)
        log_density -= torch.log(grad).sum(dim=1)

        return phi_out, x_passive, log_density


# ----------------------------------------------------------------------------------------- #
#                                                                                           #
#                                       Full layers                                         #
#                                                                                           #
# ----------------------------------------------------------------------------------------- #


class ProjectionLayer(nn.Module):
    r"""Applies the stereographic projection map S1 - {0} -> R1 to the entire
    input vector.

    The projection map is defined as

        \phi = \tan( (x - \pi) / 2 )

    And the gradient of its inverse is

        dx / d\phi = 2 / (1 + \phi^2)

    Methods
    -------
    forward(x_in, log_density)
    """

    def forward(self, x_in, log_density):
        """Forward pass of the projection transformation."""
        phi_out = torch.tan(0.5 * (x_in - pi))
        log_density -= torch.log1p(phi_out ** 2).sum(dim=2)
        return phi_out, log_density


class InverseProjectionLayer(nn.Module):
    r"""Applies the inverse stereographic projection map R1 -> S1 - {0} to the
    entire input vector.

    The inverse projection map is defined as

        \phi = 2 \arctan(x) + \pi

    And the gradient of its inverse is

        dx / d\phi = 1/2 * sec^2( (\phi - \pi) / 2 )

    Attributes
    ----------
    phase_shift: nn.Parameter
        A learnable phase shift.

    Methods
    -------
    forward(x_in, log_density)
        see docstring for anvil.layers
    """

    def __init__(self):
        super().__init__()
        self.phase_shift = nn.Parameter(torch.rand(1))

    def forward(self, x_in, log_density):
        """Forward pass of the inverse projection transformation."""
        phi_out = 2 * torch.atan(x_in) + pi
        log_density -= 2 * torch.log(torch.cos(0.5 * (phi_out - pi))).sum(dim=2)
        return (phi_out + self.phase_shift) % (2 * pi), log_density


class ProjectionLayer2D(nn.Module):
    r"""Applies the stereographic projection map S2 - {0} -> R2 to the entire
    input vector.

    The projection map is defined as

        \phi_1 = \tan(x_1 / 2) \cos(x_2)
        \phi_2 = \tan(x_1 / 2) \sin(x_2)

    where ( R = \tan(x_1 / 2), x_2 ), ( \phi_1, \phi_2 ) are, respectively, polar and
    Euclidean coordinates on R2.

    The Jacobian determinant of the inverse is

        | \det J | = 2 / ( R(1 + R^2) )

    where R is the radial coordinate defined above.

    Methods
    -------
    forward(x_in, log_density)
        see docstring for anvil.layers
    """

    def forward(self, x_in, log_density):
        """Forward pass of the projection transformation."""
        polar, azimuth = x_in.split(1, dim=1)
        rad = torch.tan(0.5 * polar)  # radial coordinate

        # -1 factor because actually want azimuth - pi, but -1 is faster than shift by pi
        phi_out = -rad * torch.cat((torch.cos(azimuth), torch.sin(azimuth)), dim=1)
        log_density -= torch.log(rad + rad.pow(3)).sum(dim=2)

        return phi_out, log_density


class InverseProjectionLayer2D(nn.Module):
    r"""Applies the inverse stereographic projection map R2 -> S2 - {0} to the
    entire input vector.

    The inverse projection map is defined, with R = \sqrt(x_1^2 + x_2^2), as

        \phi_1 = 2 \arctan(R)
        \phi_2 = \arctan2(x_1, x_2)

    where \arctan2 is a variant of \arctan than maps inputs to (-pi, pi).

    The Jacobian determinant of the projection map is

        | \det J | = 1/2 \sec^2(\phi_1 / 2) \tan(\phi_1 / 2)

    Methods
    -------
    forward(x_in, log_density)
        see docstring for anvil.layers
    """

    def forward(self, x_in, log_density):
        """Forward pass of the inverse projection transformation."""
        proj_x, proj_y = x_in.split(1, dim=1)

        polar = 2 * torch.atan(torch.sqrt(proj_x.pow(2) + proj_y.pow(2)))
        azimuth = torch.atan2(proj_x, proj_y) + pi

        phi_out = torch.cat((polar, azimuth), dim=1)
        log_density += (
            torch.log(torch.sin(0.5 * polar)) - 3 * torch.log(torch.cos(0.5 * polar))
        ).sum(dim=2)

        return phi_out, log_density


class GlobalAffineLayer(nn.Module):
    r"""Applies an affine transformation to every data point using a given scale and shift,
    which are *not* learnable. Useful to shift the domain of a learned distribution. This is
    done at the cost of a constant term in the logarithm of the Jacobian determinant, which
    is ignored.

    Parameters
    ----------
    scale: (int, float)
        Every data point will be multiplied by this factor.
    shift: (int, float)
        Every scaled data point will be shifted by this factor.

    Methods
    -------
    forward(x_in, log_density)
        see docstring for anvil.layers
    """

    def __init__(self, scale, shift):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x_in, log_density):
        """Forward pass of the global affine transformation."""
        return self.scale * x_in + self.shift, log_density
