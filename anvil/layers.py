# -*- coding: utf-8 -*-
r"""
layers.py
"""
import torch
import torch.nn as nn
from functools import partial

from anvil.core import Sequential, NeuralNetwork


class CouplingLayer(nn.Module):
    """
    Base class for coupling layers.

    A generic coupling transformation takes the form

        x_a = \phi_a
        x_b = C( \phi_b ; {N(\phi_a)} )

    where the D-dimensional \phi vector has been split into two D/2-dimensional vectors
    \phi_a and \phi_b, and {N(\phi_a)} is a set of functions, possible neural networks,
    which take \phi_a as parameters.
    
    Input and output data are flat vectors stacked in the first dimension (batch dimension).
    """

    def __init__(self, i_layer: int, size_half: int, even_sites: bool):
        super().__init__()

        if even_sites:
            # a is first half of input vector
            self._a_ind = slice(0, size_half)
            self._b_ind = slice(size_half, 2 * size_half)
            self.join_func = torch.cat
            partition = "even"
        else:
            # a is second half of input vector
            self._a_ind = slice(size_half, 2 * size_half)
            self._b_ind = slice(0, size_half)
            self.join_func = lambda a, *args, **kwargs: torch.cat(
                (a[1], a[0]), *args, **kwargs
            )
            partition = "odd"

        self.label = f"Layer: {i_layer}, partition: {partition}"

    def __str__(self):
        return f"{self.label}\n------------------------\n{super().__str__()}"


class AffineLayer(CouplingLayer):
    r"""Extension to `nn.Module` for an affine transformation layer as described
    in https://arxiv.org/abs/1904.12072.

    Affine transformation, x = g_i(\phi), defined as:

        x_a = \phi_a
        x_b = \phi_b * exp(s_i(\phi_a)) + t_i(\phi_a)

    Parameters
    ----------
    i_layer: int
        layer index, used for debugging
    size_half: int
        Half of the configuration size, which is the size of the input vector
        for the neural networks.
    s_hidden_shape: list
        list containing hidden vector sizes for s network.
    t_hidden_shape: list
        list containing hidden vector sizes for t network.
    s_activation: str
        string which is a key for an activation function for all but the final
        layer of the s network
    t_activation: str
        string which is a key for an activation function for all but the final
        layer of the t network
    s_final_activation: str
        string which is a key for an activation function for the final layer
        of the s network
    t_final_activation: str
        string which is a key for an activation function for the final layer
        of the t network
    batch_normalise: bool
        flag indicating whether or not to use batch normalising within the
        neural networks
    even_sites: bool
        dictates which half of the data is transformed as a and b, since
        successive affine transformations alternate which half of the data is
        passed through neural networks.

    Attributes
    ----------
    s_network: torch.nn.Module
        the dense layers of network s, values are intialised as per the
        default initialisation of `nn.Linear`
    t_network: torch.nn.Module
        the dense layers of network t, values are intialised as per the
        default initialisation of `nn.Linear`

    Methods
    -------
    forward(x_input, log_density):
        performs the transformation of the *inverse* coupling layer, denoted
        g_i^{-1}(x). Returns the output vector along with the contribution
        to the determinant of the jacobian of the *forward* transformation.
        = \frac{\partial g(\phi)}{\partial \phi}

    """

    def __init__(
        self,
        i_layer: int,
        size_half: int,
        *,
        s_hidden_shape=[24,],
        t_hidden_shape=[24,],
        s_activation="leaky_relu",
        t_activation="leaky_relu",
        s_final_activation="leaky_relu",
        t_final_activation=None,
        batch_normalise=False,
        even_sites: bool,
    ):
        super().__init__(i_layer, size_half, even_sites)

        # Construct networks
        self.s_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=s_hidden_shape,
            activation=s_activation,
            final_activation=s_final_activation,
            batch_normalise=batch_normalise,
            label=f"({self.label}) 's' network",
        )
        self.t_network = NeuralNetwork(
            size_in=size_half,
            size_out=size_half,
            hidden_shape=t_hidden_shape,
            activation=t_activation,
            final_activation=t_final_activation,
            batch_normalise=batch_normalise,
            label=f"({self.label}) 't' network",
        )

    def forward(self, x_input, log_density) -> torch.Tensor:
        r"""performs the transformation of the inverse coupling layer, denoted
        g_i^{-1}(x)

        inverse transformation, \phi = g_i^{-1}(x), defined as:

        \phi_a = x_a
        \phi_b = (x_b - t_i(x_a)) * exp(-s_i(x_a))

        see eq. (10) of https://arxiv.org/pdf/1904.12072.pdf

        Also computes the logarithm of the jacobian determinant for the
        forward transformation (inverse of the above), which is equal to
        the logarithm of

        \frac{\partial g(\phi)}{\partial \phi} = prod_j exp(s_i(\phi)_j)

        Parameters
        ----------
        x_input: torch.Tensor
            stack of vectors x, shape (N_states, D)
        log_density: torch.Tensor
            current value for the logarithm of the volume element for the density
            defined by the map.

        Returns
        -------
        out: torch.Tensor
            stack of transformed vectors phi, with same shape as input
        log_density: torch.Tensor
            updated log density for the map, with the addition of the logarithm of
            the jacobian determinant for the inverse of the transformation applied here.
        """
        x_a = x_input[..., self._a_ind]
        x_b = x_input[..., self._b_ind]
        s_out = self.s_network(x_a)
        t_out = self.t_network(x_a)
        phi_b = (x_b - t_out) * torch.exp(-s_out)

        phi_out = self.join_func([x_a, phi_b], dim=-1)
        log_density += s_out.sum(dim=tuple(range(1, len(s_out.shape)))).view(-1, 1)

        return phi_out, log_density


def affine_transformation(i_layer, size_half, layer_spec={}):
    """Action which returns a callable object that performs an affine coupling
    transformation on both even and odd lattice sites."""
    coupling_transformation = partial(AffineLayer, i_layer, size_half, **layer_spec)
    return Sequential(
        coupling_transformation(even_sites=True),
        coupling_transformation(even_sites=False),
    )


LAYER_OPTIONS = {"real_nvp": affine_transformation}
