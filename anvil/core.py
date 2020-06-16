r"""
coupling.py
"""
import torch
import torch.nn as nn
from itertools import cycle

from reportengine import collect

from math import pi

ACTIVATION_LAYERS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "celu": nn.CELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    None: nn.Identity,
}


class Sequential(nn.Sequential):
    """Modify the nn.Sequential class so that it takes an input vector *and* a
    value for the current logarithm of the model density, returning an output
    vector and the updated log density."""

    def forward(self, x, log_dens):
        for module in self:
            x, log_dens = module(x, log_dens)
        return x, log_dens


class NeuralNetwork(nn.Module):
    """Generic class for neural networks used in coupling layers.

    Networks consist of 'blocks' of
        - Dense (linear) layer
        - Batch normalisation layer
        - Activation function

    Parameters
    ----------
    size_in: int
        Number of nodes in the input layer
    size_out: int
        Number of nodes in the output layer
    hidden_shape: list
        List specifying the number of nodes in the intermediate layers
    activation: (str, None)
        Key representing the activation function used for each layer
        except the final one.
    final_activation: (str, None)
        Key representing the activation function used on the final
        layer.
    batch_normalise: bool
        Flag dictating whether batch normalisation should be performed
        before the activation function.

    Methods
    -------
    forward:
        The forward pass of the network, mapping a batch of input vectors
        with 'size_in' nodes to a batch of output vectors of 'size_out'
        nodes.
    """

    def __init__(
        self,
        *,
        size_in: int,
        size_out: int,
        hidden_shape: list,
        activation: (str, None),
        final_activation: (str, None) = None,
        batch_normalise: bool = False,
    ):
        super(NeuralNetwork, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.hidden_shape = hidden_shape

        if batch_normalise:
            self.batch_norm = nn.BatchNorm1d
        else:
            self.batch_norm = nn.Identity

        self.activation_func = ACTIVATION_LAYERS[activation]
        self.final_activation_func = ACTIVATION_LAYERS[final_activation]

        # nn.Sequential object containing the network layers
        self.network = self._construct_network()

    def _block(self, f_in, f_out, activation_func):
        """Constructs a single 'dense block' which maps 'f_in' inputs to
        'f_out' output features. Returns a list with three elements:
            - Dense (linear) layer
            - Batch normalisation (or identity if this is switched off)
            - Activation function
        """
        return [
            nn.Linear(f_in, f_out),
            self.batch_norm(f_out),
            activation_func(),
        ]

    def _construct_network(self):
        """Constructs the neural network from multiple calls to _block.
        Returns a torch.nn.Sequential object which has the 'forward' method built in.
        """
        layers = self._block(self.size_in, self.hidden_shape[0], self.activation_func)
        for f_in, f_out in zip(self.hidden_shape[:-1], self.hidden_shape[1:]):
            layers += self._block(f_in, f_out, self.activation_func)
        layers += self._block(
            self.hidden_shape[-1], self.size_out, self.final_activation_func
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        """Forward pass of the network."""
        return self.network(x)


class CoupleRedBlack(nn.Module):
    """Generic class for sequences of coupling transformations which couple different
    lattice sites.

    Lattice sites are partitioned into two groups: 'red' and 'black', as per a
    checkerboard pattern. A coupling layer transforms either the red or the black sites,
    using the passive partition as input for a (set of) neural networks which
    parameterise the transformation.

    Parameters
    ----------
    coupling_layer: nn.Module
        A nn.Module from anvil.layers which implements a single coupling transformation.
    n_lattice: int
        Number of sites on the lattice.
    n_pairs: int
        Number of pairs of coupling transformations, which is the number of times each
        data point is transformed.
    layer_spec: dict
        A dictionary containing keyword arguments for `coupling_layer`.
    
    Attributes
    ----------
    red_layers: nn.ModuleList
        A list of `n_pairs` instances of `coupling_layer`, which will transform the red
        partition whilst taking the black partition as parameters.
    black_layers: nn.ModuleList
        As above for the black partition.

    Methods
    -------
    forward(x_in, log_density)
        Takes a tensor of input data with dimensions (n_batch, n_components, n_lattice),
        and a tensor of the current logarithm of the probability density function, with
        dimensions (n_batch, 1). Returns a tensor of transformed data and an updated
        log density.
    """

    def __init__(
        self, coupling_layer, n_lattice: int, layer_spec: dict, *, n_couple=1,
    ):
        super().__init__()
        self.lattice_half = n_lattice // 2

        self.red_layers = nn.ModuleList(
            [
                coupling_layer(self.lattice_half, self.lattice_half, **layer_spec)
                for _ in range(n_couple)
            ]
        )
        self.black_layers = nn.ModuleList(
            [
                coupling_layer(self.lattice_half, self.lattice_half, **layer_spec)
                for _ in range(n_couple)
            ]
        )

    def forward(self, x_in, log_density):
        """Forward pass of the sequence of coupling transformations."""
        x_r, x_b = x_in.squeeze(dim=1).split(self.lattice_half, dim=1)

        for red_layer, black_layer in zip(self.red_layers, self.black_layers):
            x_r, log_density = red_layer(x_r, x_b, log_density)
            x_b, log_density = black_layer(x_b, x_r, log_density)

        phi_out = torch.cat((x_r, x_b), dim=1).unsqueeze(dim=1)
        return phi_out, log_density


class CoupleRedBlackNd(nn.Module):
    """N-dimensional generalisation of CoupleRedBlack, which couples the red/black sites
    for multiple components, each set of components separately."""

    def __init__(
        self, coupling_layers: list, n_lattice: int, layer_spec: dict, *, n_couple=1
    ):
        super().__init__()

        self.component_layers = nn.ModuleList(
            [
                CoupleRedBlack(coupling_layer, n_lattice, layer_spec, n_couple=n_couple)
                for coupling_layer in coupling_layers
            ]
        )

    def forward(self, x_in, log_density):
        x_components = x_in.split(1, dim=1)

        phi_out = []
        for x_i, layer in zip(x_components, self.component_layers):
            phi_i, log_density = layer(x_i, log_density)
            phi_out.append(phi_i)

        return torch.cat(phi_out, dim=1), log_density


class CoupleComponents(nn.Module):
    """Couple each set of components with one other set."""

    def __init__(self, coupling_layers: list, n_lattice: int, layer_spec: dict):
        super().__init__()
        self.n_components = len(coupling_layers)

        self.component_layers = nn.ModuleList(
            [
                coupling_layer(n_lattice, n_lattice, **layer_spec)
                for coupling_layer in coupling_layers
            ]
        )

    def forward(self, x_in, log_density):
        x_components = x_in.split(1, dim=1)

        phi_out = []
        for i, layer in enumerate(self.component_layers):
            x_in = x_components[i].squeeze(dim=1)
            x_passive = x_components[(i + 1) % self.n_components].squeeze(dim=1)
            phi_i, log_density = layer(x_in, x_passive, log_density)
            phi_out.append(phi_i.unsqueeze(dim=1))

        return torch.cat(phi_out, dim=1), log_density


class ConvexCombination(nn.Module):
    r"""
    Class which takes a set of normalising flows and constructs a convex combination
    of their outputs to produce a single output distribution and the logarithm of its
    volume element, calculated using the change of variables formula.

    A convex combination is a weighted sum of elements
        
        f(x_1, x_2, ..., x_N) = \sum_{i=1}^N \rho_i x_i

    where the weights are normalised, \sum_{i=1}^N \rho_i = 1.

    Parameters
    ----------
    flow_replica
        A list of replica normalising flow models.
    
    Methods
    -------
    forward(x_input, log_density):
        Returns the convex combination of probability densities output by the flow
        replica, along with the convex combination of logarithms of probability 
        densities.

    Notes
    -----
    It is assumed that the log_density input to the forward method is the logarithm
    of a *normalised* probability density - i.e. the base log density is normalised and
    we don't neglect additive constants to the log density during the flow.
    """

    def __init__(self, flow_replica):
        super().__init__()
        self.flows = nn.ModuleList(flow_replica)
        self.weights = nn.Parameter(torch.rand(len(flow_replica)))
        self.norm_func = nn.Softmax(dim=0)

    def forward(self, x_input, log_density_base):
        """Forward pass of the model.

        Parameters
        ----------
        x_input: torch.Tensor
            stack of input vectors drawn from the base distribution
        log_density_base: torch.Tensor
            The logarithm of the probability density of the base distribution.

        Returns
        -------
        out: torch.Tensor
            the convex combination of the output probability densities.
        log_density: torch.Tensor
            the logarithm of the probability density for the convex combination of
            output densities, added to the base log density.
        """
        weights_norm = self.norm_func(self.weights)

        phi_out, density = 0, 0
        for weight, flow in zip(weights_norm, self.flows):
            # don't want each flow to update same input tensor
            input_copy = x_input.clone()
            # don't want to add to base density
            zero_density = torch.zeros_like(log_density_base)

            phi_flow, log_dens_flow = flow(input_copy, zero_density)
            phi_out += weight * phi_flow
            density += weight * torch.exp(log_dens_flow)

        return phi_out, log_density_base + torch.log(density)


_normalising_flow = collect("model_action", ("model_spec",))


def normalising_flow(_normalising_flow, i_mixture=1):
    """Return a callable model which is a normalising flow constructed via
    function composition."""
    return _normalising_flow[0]


_flow_replica = collect("normalising_flow", ("mixture_indices",))


def convex_combination(_flow_replica):
    """Return a callable model which is a convex combination of multiple
    normalising flows."""
    return ConvexCombination(_flow_replica)
