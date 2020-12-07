# -*- coding: utf-8 -*-
r"""
layers.py

Contains nn.Modules which implement transformations of an input vector whilst computing the
Jacobian determinant of the inverse transformation.

Each transformation layers may contain several neural networks or learnable parameters.

A normalising flow, f, can be constructed from multiple layers using function composition:

        f(\phi) = g_1( g_2( ... ( g_n( \phi ) ) ... ) )

which is implemented using the architecture provided by torch.nn


All layers in this module contain a `forward` method which takes two torch.tensor objects
as inputs:

    - a batch of input vectors x, dimensions (N_batch, D) where D is the total number of
      units being transformed (i.e. number of lattice sites multiplied by field dimension).

    - a batch of scalar values for the logarithm of the 'current' probability density, at
      this stage in the normalising flow.

and returns two torch.tensor objects:

    - a batch of vectors \phi which have been transformed according to the *inverse*
    transformation g_i^{-1}, with the same dimensions as the input.

    - the updated logarithm of the probability density, including the contribution from
      this transformation, | \det \partial g_i / \partial \phi |.
"""
import torch
import torch.nn as nn
from torchsearchsorted import searchsorted
from math import pi

from anvil.core import NeuralNetwork

import numpy as np


class CouplingLayer(nn.Module):
    """
    Base class for coupling layers.

    A generic coupling transformation takes the form

        x_a = \phi_a
        x_b = C( \phi_b ; {N(\phi_a)} )

    where the D-dimensional \phi vector has been split into two D/2-dimensional vectors
    \phi_a and \phi_b, and {N(\phi_a)} is a set of functions, possible neural networks,
    which take \phi_a as parameters.

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.


    Attributes
    ----------
    a_ind: slice (protected)
        Slice object which can be used to access the passive partition.
    b_ind: slice (protected)
        Slice object which can be used to access the partition that gets transformed.
    join_func: function (protected)
        Function which returns the concatenation of the two partitions in the
        appropriate order.
    """

    def __init__(self, size_half: int, even_sites: bool):
        super().__init__()

        if even_sites:
            # a is first half of input vector
            self._a_ind = slice(0, size_half)
            self._b_ind = slice(size_half, 2 * size_half)
            self._join_func = torch.cat
        else:
            # a is second half of input vector
            self._a_ind = slice(size_half, 2 * size_half)
            self._b_ind = slice(0, size_half)
            self._join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )


class AdditiveLayer(CouplingLayer):
    def __init__(
        self,
        i: int,
        size_half: int,
        *,
        hidden_shape: list,
        activation: str,
        symmetric: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.i = i

        self.t_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            symmetric=symmetric,
        )

        self.symmetric = symmetric

    def forward(self, x_input, log_density) -> torch.Tensor:
        r"""Forward pass of affine transformation."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]
        if self.symmetric:
            x_a_stand = x_a / x_a.std()
        else:
            x_a_stand = (
                x_a  # (x_a - x_a.mean()) / x_a.std()  # reduce numerical instability
            )
        t_out = self.t_network(x_a_stand)

        phi_b = x_b - t_out

        phi_out = self._join_func([x_a, phi_b], dim=1)

        if phi_out.requires_grad is False:
            np.savetxt(f"layer_{self.i}.txt", phi_out)

        return phi_out, log_density


class AffineLayer(CouplingLayer):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    An affine transformation, x = g_i(\phi), is defined as:

        x_a = \phi_a
        x_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector for the
        neural networks.
    hidden_shape: list
        list containing hidden vector sizes for the neural networks.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the networks.
    s_final_activation: str
        string which is a key for an activation function, which the output of the s
        network will be passed through.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since successive
        affine transformations alternate which half of the data is passed through
        neural networks.

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
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(
        self,
        i: int,
        size_half: int,
        *,
        hidden_shape: list,
        activation: str,
        s_final_activation: str,
        symmetric_networks: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.i = i

        self.s_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=s_final_activation,
            symmetric=symmetric_networks,
        )
        self.t_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            symmetric=symmetric_networks,
        )
        # NOTE: Could potentially have non-default inputs for s and t networks
        # by adding dictionary of overrides - e.g. s_options = {}

        self.symmetric_networks = symmetric_networks

    def forward(self, x_input, log_density, neg) -> torch.Tensor:
        r"""Forward pass of affine transformation."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]
        if self.symmetric_networks:
            x_a_stand = x_a / x_a.std()
        else:
            x_a_stand = (x_a - x_a.mean()) / x_a.std()  # reduce numerical instability

            x_a_stand[neg] = -x_a_stand[neg]  # still symmetric, but different approach

        s_out = self.s_network(x_a_stand)
        t_out = self.t_network(x_a_stand)

        if self.symmetric_networks:
            s_out.abs_()
        else:
            t_out[neg] = -t_out[neg]

        phi_b = (x_b - t_out) * torch.exp(-s_out)

        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density += s_out.sum(dim=1, keepdim=True)

        if False:  # phi_out.requires_grad is False:
            np.savetxt(f"layer_{self.i}.txt", phi_out)

        return phi_out, log_density


class NCPLayer(CouplingLayer):
    r"""A coupling layer which performs a transformation from (0, 2\pi) -> (0, 2\pi),
    which is an affine transformation X -> \alpha * X + \beta on the Euclidean vector
    obtained by stereographic projection of the input data.

    The transformation x = g(\phi) is defined as

        x_a = \phi_a
        x_b = (
                2 \arctan( \alpha(\phi_a) + \tan((\phi_b - \pi) / 2) + \beta(\phi_a))
                + \pi + \theta
              )

    where \alpha and \beta are neural networks, and \theta is a learnable global phase shift.

    The Jacobian determinant can be computed analytically for the full transformation,
    including projection and inverse.

        | \det J | = \prod_n (
            (1 + \beta ** 2) / \alpha * \sin^2(x / 2) + \beta
            + \alpha * \cos^2(x / 2)
            - \beta * \sin(x_b)
            )

    Attributes
    ----------
    s_network: NeuralNetwork
        When exponentiated, this is the linear transformation \alpha.
    t_network: NeuralNetwork
        The shift part (\beta) of the affine transformation.
    phase_shift: nn.Parameter
        A learnable global phase shift.

    Notes
    -----
    The Jacobian determinant is computed using the gradient of the inverse transformation,
    which is the reciprocal of the gradient of the forward transformation.

    This can be thought of as a lightweight version of real_nvp_circle, with just one affine
    transformation.
    """

    def __init__(
        self,
        size_half: int,
        *,
        hidden_shape: list,
        activation: str,
        s_final_activation: str,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)

        self.s_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=s_final_activation,
        )
        self.t_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
        )
        self.phase_shift = nn.Parameter(torch.rand(1))

    def forward(self, x_input, log_density):
        """Forward pass of the project-affine-inverse transformation."""
        x_a = x_input[..., self._a_ind]
        x_b = x_input[..., self._b_ind]
        x_a_stand = (x_a - x_a.mean()) / x_a.std()  # reduce numerical instability

        alpha = torch.exp(self.s_network(x_a_stand))
        beta = self.t_network(x_a_stand)

        phi_b = (
            2 * torch.atan(alpha * torch.tan((x_b - pi) / 2) + beta)
            + pi
            + self.phase_shift
        ) % (2 * pi)

        log_density += torch.log(
            (1 + beta ** 2) / alpha * torch.sin(x_b / 2) ** 2
            + alpha * torch.cos(x_b / 2) ** 2
            - beta * torch.sin(x_b)
        ).sum(dim=1, keepdim=True)

        phi_out = self._join_func([x_a, phi_b], dim=-1)

        return phi_out, log_density


class LinearSplineLayer(CouplingLayer):
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
    size_half: int
        Half of the configuration size, which is the size of the input vector for the
        neural network.
    hidden_shape: list
        list containing hidden vector sizes the neural network.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the network.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since successive
        affine transformations alternate which half of the data is passed through
        neural network.

    Attributes
    ----------
    network: torch.nn.Module
        the dense layers of network h, values are intialised as per the default
        initialisation of `nn.Linear`

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(
        self,
        size_half: int,
        *,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.size_half = size_half
        self.n_segments = n_segments
        self.width = 1 / n_segments

        eps = 1e-6  # prevent rounding error which causes sorting into -1th bin
        self.x_knot_points = torch.linspace(-eps, 1 + eps, n_segments + 1).view(1, -1)

        self.network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half * n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=activation,
        )
        self.norm_func = nn.Softmax(dim=2)

    def forward(self, x_input, log_density):
        """Forward pass of the linear spline layer."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]

        net_out = self.norm_func(
            self.network(x_a - 0.5).view(-1, self.size_half, self.n_segments)
        )
        phi_knot_points = torch.cat(
            (
                torch.zeros(net_out.shape[0], self.size_half, 1),
                torch.cumsum(net_out, dim=2),
            ),
            dim=2,
        )

        # Sort x_b into the appropriate bin
        # NOTE: need to make x_b contiguous, otherwise searchsorted returns nonsense
        k_ind = searchsorted(self.x_knot_points, x_b.contiguous()) - 1
        k_ind.unsqueeze_(dim=-1)

        p_k = torch.gather(net_out, 2, k_ind)
        alpha = (x_b.unsqueeze(dim=-1) - k_ind * self.width) / self.width
        phi_km1 = torch.gather(phi_knot_points, 2, k_ind)

        phi_b = (phi_km1 + alpha * p_k).squeeze()
        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density -= torch.log(p_k).sum(dim=1)

        return phi_out, log_density


class QuadraticSplineLayer(CouplingLayer):
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
    size_half: int
        Half of the configuration size, which is the size of the input vector for the
        neural network.
    hidden_shape: list
        list containing hidden vector sizes the neural network.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the network.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since successive
        affine transformations alternate which half of the data is passed through
        neural network.

    Attributes
    ----------
    h_network: torch.nn.Module
        the dense layers of network h, values are intialised as per the default
        initialisation of `nn.Linear`

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(
        self,
        size_half: int,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.size_half = size_half
        self.n_segments = n_segments

        self.network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half * (2 * n_segments + 1),
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=activation,
        )
        self.w_norm_func = nn.Softmax(dim=2)

        self.eps = 1e-6  # prevent rounding error which causes sorting into -1th bin

    @staticmethod
    def h_norm_func(h_raw, w_norm):
        """Normalisation function for height values."""
        return torch.exp(h_raw) / (
            0.5 * w_norm * (torch.exp(h_raw[..., :-1]) + torch.exp(h_raw[..., 1:]))
        ).sum(dim=2, keepdim=True)

    def forward(self, x_input, log_density):
        """Forward pass of the quadratic spline layer."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]

        h_raw, w_raw = (
            self.network(x_a - 0.5)
            .view(-1, self.size_half, 2 * self.n_segments + 1)
            .split((self.n_segments + 1, self.n_segments), dim=2)
        )
        w_norm = self.w_norm_func(w_raw)
        h_norm = self.h_norm_func(h_raw, w_norm)

        x_knot_points = torch.cat(
            (
                torch.zeros(h_norm.shape[0], self.size_half, 1) - self.eps,
                torch.cumsum(w_norm, dim=2),
            ),
            dim=2,
        )
        phi_knot_points = torch.cat(
            (
                torch.zeros(h_norm.shape[0], self.size_half, 1),
                torch.cumsum(
                    0.5 * w_norm * (h_norm[..., :-1] + h_norm[..., 1:]),
                    dim=2,
                ),
            ),
            dim=2,
        )

        # Temporarily mix batch and lattice dimensions so that the bisection search
        # can be done in a single operation
        k_ind = (
            searchsorted(
                x_knot_points.contiguous().view(-1, self.n_segments + 1),
                x_b.contiguous().view(-1, 1),
            )
            - 1
        ).view(-1, self.size_half, 1)

        w_k = torch.gather(w_norm, 2, k_ind)
        h_k = torch.gather(h_norm, 2, k_ind)
        h_kp1 = torch.gather(h_norm, 2, k_ind + 1)

        x_km1 = torch.gather(x_knot_points, 2, k_ind)
        phi_km1 = torch.gather(phi_knot_points, 2, k_ind)

        alpha = (x_b.unsqueeze(dim=-1) - x_km1) / w_k
        phi_b = (
            phi_km1 + alpha * h_k * w_k + 0.5 * alpha.pow(2) * (h_kp1 - h_k) * w_k
        ).squeeze()

        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density -= torch.log(h_k + alpha * (h_kp1 - h_k)).sum(dim=1)

        return phi_out, log_density


class RationalQuadraticSplineLayer(CouplingLayer):
    r"""A coupling transformation from a finite interval to itself based on a piecewise
    rational quadratic spline function.

    The interval is divided into K segments (bins) with widths w_k and heights h_k. The
    'knot points' (\phi_k, x_k) are the cumulative sum of (h_k, w_k), starting at (-B, -B)
    and ending at (B, B).

    In addition to the w_k and h_k, the derivatives d_k at the internal knot points are
    generated by a neural network. d_0 and d_K are set to 1.

    Defing the slopes s_k = h_k / w_k and fractional position within a bin

            alpha(x) = (x - x_{k-1}) / w_k

    the coupling transformation is defined piecewise by

            \phi = C^{-1}(x, {h_k, s_k, d_k})
                 = \phi_{k-1}
                 + ( h_k(s_k * \alpha^2 + d_k * \alpha * (1 - \alpha)) )
                 / ( s_k + (d_{k+1} + d_k - 2s_k) * \alpha * (1 - \alpha) )

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector for the
        neural network.
    hidden_shape: list
        list containing hidden vector sizes the neural network.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the network.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since successive
        affine transformations alternate which half of the data is passed through
        neural network.

    Attributes
    ----------
    network: torch.nn.Module
        the dense layers of the network, values are intialised as per the default
        initialisation of `nn.Linear`

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(
        self,
        i,
        size_half: int,
        interval: int,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        symmetric_spline: bool,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.i = i
        self.j = int(even_sites)

        self.size_half = size_half
        self.n_segments = n_segments

        self.network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half * (3 * n_segments - 1),
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            symmetric=False,  # biases very useful
        )

        self.norm_func = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()

        self.B = interval
        self.eps = 1e-6

        self.force_symmetry = symmetric_spline

    def forward(self, x_input, log_density, neg):
        """Forward pass of the rational quadratic spline layer."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]
        x_a_stand = (x_a - x_a.mean()) / x_a.std()  # reduce numerical instability

        if self.force_symmetry:
            x_a_stand[neg] = -x_a_stand[neg]

        phi_b = torch.zeros_like(x_b)
        grad = torch.ones_like(x_b).unsqueeze(dim=-1)

        # Apply mask for linear tails
        inside_mask = abs(x_b) <= self.B
        x_b_in = x_b[inside_mask]
        phi_b[~inside_mask] = x_b[~inside_mask]

        h_raw, w_raw, d_raw = (
            self.network(x_a_stand)
            .view(-1, self.size_half, 3 * self.n_segments - 1)
            .split(
                (self.n_segments, self.n_segments, self.n_segments - 1),
                dim=2,
            )
        )

        if self.force_symmetry:
            h_raw[neg] = torch.flip(h_raw[neg], dims=(2,))
            w_raw[neg] = torch.flip(w_raw[neg], dims=(2,))
            d_raw[neg] = torch.flip(d_raw[neg], dims=(2,))

        h_norm = self.norm_func(h_raw[inside_mask]) * 2 * self.B
        w_norm = self.norm_func(w_raw[inside_mask]) * 2 * self.B
        d_pad = nn.functional.pad(
            self.softplus(d_raw)[inside_mask], (1, 1), "constant", 1
        )

        x_knot_points = (
            torch.cat(
                (
                    torch.zeros(w_norm.shape[0], 1) - self.eps,
                    torch.cumsum(w_norm, dim=1),
                ),
                dim=1,
            )
            - self.B
        )
        phi_knot_points = (
            torch.cat(
                (
                    torch.zeros(h_norm.shape[0], 1),
                    torch.cumsum(h_norm, dim=1),
                ),
                dim=1,
            )
            - self.B
        )

        k_ind = (
            searchsorted(
                x_knot_points.contiguous(),
                x_b_in.contiguous().view(-1, 1),
            )
            - 1
        ).clamp(0, self.n_segments - 1)

        w_k = torch.gather(w_norm, 1, k_ind)
        h_k = torch.gather(h_norm, 1, k_ind)
        s_k = h_k / w_k
        d_k = torch.gather(d_pad, 1, k_ind)
        d_kp1 = torch.gather(d_pad, 1, k_ind + 1)

        x_km1 = torch.gather(x_knot_points, 1, k_ind)
        phi_km1 = torch.gather(phi_knot_points, 1, k_ind)

        alpha = (x_b_in.unsqueeze(dim=-1) - x_km1) / w_k

        phi_b[inside_mask] = (
            phi_km1
            + (h_k * (s_k * alpha.pow(2) + d_k * alpha * (1 - alpha)))
            / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha))
        ).squeeze()

        grad[inside_mask] = (
            s_k.pow(2)
            * (
                d_kp1 * alpha.pow(2)
                + 2 * s_k * alpha * (1 - alpha)
                + d_k * (1 - alpha).pow(2)
            )
        ) / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha)).pow(2)

        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density -= torch.log(grad).sum(dim=1)

        if False:  # phi_out.requires_grad is False:

            np.savetxt(f"layer_{self.i}.txt", phi_out)
            np.savetxt(f"x_kp_{self.i}.txt", x_knot_points)
            np.savetxt(f"phi_kp_{self.i}.txt", phi_knot_points)
            np.savetxt("h.txt", h_norm[0:4, :])
            np.savetxt("w.txt", w_norm[0:4, :])
            np.savetxt("d.txt", d_pad[0:4, :])

        return phi_out, log_density


class CircularSplineLayer(CouplingLayer):
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
    size_half: int
        Half of the configuration size, which is the size of the input vector for the
        neural network.
    hidden_shape: list
        list containing hidden vector sizes the neural network.
    activation: str
        string which is a key for an activation function for all but the final layers
        of the network.
    even_sites: bool
        dictates which half of the data is transformed as a and b, since successive
        affine transformations alternate which half of the data is passed through
        neural network.

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
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(
        self,
        i,
        size_half: int,
        n_segments: int,
        hidden_shape: list,
        activation: str,
        even_sites: bool,
    ):
        super().__init__(size_half, even_sites)
        self.i = i
        self.j = int(even_sites)

        self.size_half = size_half
        self.n_segments = n_segments

        self.network = NeuralNetwork(
            size_in=2 * size_half,
            size_out=size_half * 3 * n_segments,
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=None,
            symmetric=False,
        )
        self.phase_shift = nn.Parameter(torch.rand(1))

        self.norm_func = nn.Softmax(dim=2)
        self.softplus = nn.Softplus()

        self.eps = 1e-6

    def forward(self, x_input, log_density, neg):
        """Forward pass of the rational quadratic spline layer."""
        x_a = x_input[:, self._a_ind]
        x_b = x_input[:, self._b_ind]

        x_for_net = torch.cat(
            [
                torch.cos(x_a),
                torch.sin(x_a),
            ],
            dim=1,
        )

        h_raw, w_raw, d_raw = (
            self.network(x_for_net)
            .view(-1, self.size_half, 3 * self.n_segments)
            .split((self.n_segments, self.n_segments, self.n_segments), dim=2)
        )
        h_norm = self.norm_func(h_raw) * 2 * pi
        w_norm = self.norm_func(w_raw) * 2 * pi
        d_norm = self.softplus(d_raw)

        x_knot_points = torch.cat(
            (
                torch.zeros(w_norm.shape[0], self.size_half, 1) - self.eps,
                torch.cumsum(w_norm, dim=2),
            ),
            dim=2,
        )
        phi_knot_points = torch.cat(
            (
                torch.zeros(h_norm.shape[0], self.size_half, 1),
                torch.cumsum(h_norm, dim=2),
            ),
            dim=2,
        )

        k_ind = (
            (
                searchsorted(
                    x_knot_points.contiguous().view(-1, self.n_segments + 1),
                    x_b.contiguous().view(-1, 1),
                )
                - 1
            )
            .view(-1, self.size_half, 1)
            .clamp(0, self.n_segments - 1)
        )

        w_k = torch.gather(w_norm, 2, k_ind)
        h_k = torch.gather(h_norm, 2, k_ind)
        s_k = h_k / w_k
        d_k = torch.gather(d_norm, 2, k_ind)
        d_kp1 = torch.gather(d_norm, 2, (k_ind + 1) % self.n_segments)

        x_km1 = torch.gather(x_knot_points, 2, k_ind)
        phi_km1 = torch.gather(phi_knot_points, 2, k_ind)

        alpha = (x_b.unsqueeze(dim=-1) - x_km1) / w_k

        phi_b = (
            phi_km1
            + (h_k * (s_k * alpha.pow(2) + d_k * alpha * (1 - alpha)))
            / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha))
        ).squeeze()
        phi_b = (phi_b + self.phase_shift) % (2 * pi)

        grad = (
            s_k.pow(2)
            * (
                d_kp1 * alpha.pow(2)
                + 2 * s_k * alpha * (1 - alpha)
                + d_k * (1 - alpha).pow(2)
            )
        ) / (s_k + (d_kp1 + d_k - 2 * s_k) * alpha * (1 - alpha)).pow(2)

        phi_out = self._join_func([x_a, phi_b], dim=1)
        log_density -= torch.log(grad).sum(dim=1)
        
        if phi_out.requires_grad is False:
            #np.savetxt(f"layer_{self.i}.txt", phi_out)
            np.savetxt(f"x_kp_{self.i}.txt", x_knot_points.view(-1, self.n_segments + 1))
            np.savetxt(f"phi_kp_{self.i}.txt", phi_knot_points.view(-1, self.n_segments + 1))
            np.savetxt("h.txt", h_norm.view(-1, self.n_segments)[0:4, :])
            np.savetxt("w.txt", w_norm.view(-1, self.n_segments)[0:4, :])
            np.savetxt("d.txt", d_norm.view(-1, self.n_segments)[0:4, :])

        return phi_out, log_density


class ProjectionLayer(nn.Module):
    r"""Applies the stereographic projection map S1 - {0} -> R1 to the entire
    input vector.

    The projection map is defined as

        \phi = \tan( (x - \pi) / 2 )

    And the gradient of its inverse is

        dx / d\phi = 2 / (1 + \phi^2)

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def forward(self, x_input, log_density):
        """Forward pass of the projection transformation."""
        phi_out = torch.tan(0.5 * (x_input - pi))
        log_density -= torch.log1p(phi_out ** 2).sum(dim=1, keepdim=True)
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
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self):
        super().__init__()
        self.phase_shift = nn.Parameter(torch.rand(1))

    def forward(self, x_input, log_density):
        """Forward pass of the inverse projection transformation."""
        phi_out = 2 * torch.atan(x_input) + pi
        log_density -= 2 * torch.log(torch.cos(0.5 * (phi_out - pi))).sum(
            dim=1, keepdim=True
        )
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

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self, size_half: int):
        super().__init__()
        self.size_half = size_half
        self.size_out = 2 * size_half

    def forward(self, x_input, log_density):
        """Forward pass of the projection transformation."""
        polar, azimuth = x_input.view(-1, self.size_half, 2).split(1, dim=2)
        rad = torch.tan(0.5 * polar)  # radial coordinate

        # -1 factor because actually want azimuth - pi, but -1 is faster than shift by pi
        phi_out = -rad * torch.cat((torch.cos(azimuth), torch.sin(azimuth)), dim=2)
        log_density -= torch.log(rad + rad.pow(3)).sum(dim=1)

        return phi_out.view(-1, self.size_out), log_density


class InverseProjectionLayer2D(nn.Module):
    r"""Applies the inverse stereographic projection map R2 -> S2 - {0} to the
    entire input vector.

    The inverse projection map is defined, with R = \sqrt(x_1^2 + x_2^2), as

        \phi_1 = 2 \arctan(R)
        \phi_2 = \arctan2(x_1, x_2)

    where \arctan2 is a variant of \arctan than maps inputs to (-pi, pi).

    The Jacobian determinant of the projection map is

        | \det J | = 1/2 \sec^2(\phi_1 / 2) \tan(\phi_1 / 2)

    Parameters
    ----------
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self, size_half: int):
        super().__init__()
        self.size_half = size_half
        self.size_out = 2 * size_half

    def forward(self, x_input, log_density):
        """Forward pass of the inverse projection transformation."""
        proj_x, proj_y = x_input.view(-1, self.size_half, 2).split(1, dim=2)

        polar = 2 * torch.atan(torch.sqrt(proj_x.pow(2) + proj_y.pow(2)))
        azimuth = torch.atan2(proj_x, proj_y) + pi

        phi_out = torch.cat((polar, azimuth), dim=2)
        log_density += (
            torch.log(torch.sin(0.5 * polar)) - 3 * torch.log(torch.cos(0.5 * polar))
        ).sum(dim=1)

        return phi_out.view(-1, self.size_out), log_density


class GlobalAdditiveLayer(nn.Module):
    def __init__(self, shift_init=0.0, learnable=True):
        super().__init__()

        if learnable:
            self.shift = nn.Parameter(torch.tensor([shift_init]))
            self.F = nn.Softplus()
        else:
            self.shift = shift_init
            self.F = lambda x: x

    def forward(self, x_input, log_density, neg):
        shift = x_input.mean(dim=1, keepdim=True).sign() * self.F(self.shift)
        return x_input + shift, log_density


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
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self, scale=1, shift=0):
        super().__init__()
        if scale < 0:
            self.scale = nn.Parameter(torch.tensor([1.0]))
        else:
            self.scale = scale
        self.softplus = nn.Softplus()

        self.shift = shift

    def forward(self, x_input, log_density):
        """Forward pass of the global affine transformation."""
        gamma = self.softplus(self.scale)
        # print(gamma)
        log_density -= torch.log(gamma) * x_input.shape[1]
        return gamma * x_input + self.shift, log_density


class BatchNormLayer(nn.Module):
    """Performs batch normalisation on the input vector.

    Unlike traditional batch normalisation, due to translational invariance we take
    averages over each dimension as well as the batch and scale every dimension by the
    same quantity.

    In addition, there is the option for a pre-defined or learnable multiplicative
    factor, to be applied after the batch normalisation.

    Parameters
    ----------
    scale: int
        An additional scale factor to be applied after batch normalisation. A negative
        input means this will be a learnable parameter, initialised at 1.

    Methods
    -------
    forward(x_input, log_density)
        see docstring for anvil.layers
    """

    def __init__(self, scale=1, learnable=False):
        super().__init__()
        self.soft = nn.Softplus()
        Fm1 = lambda g: torch.log(torch.exp(g) - 1)

        if learnable:
            self.scale = nn.Parameter(Fm1(torch.tensor([scale])))
        else:
            self.scale = Fm1(torch.tensor([scale]))

        self.eps = 0.00001

    def forward(self, x_input, log_density, neg):
        """Forward pass of the batch normalisation transformation."""
        gamma = self.soft(self.scale)
        mult = gamma / torch.sqrt(torch.var(x_input) + self.eps)
        phi_out = (x_input - x_input.mean()) * mult
        log_density -= x_input.shape[1] * torch.log(mult)
        return phi_out, log_density
