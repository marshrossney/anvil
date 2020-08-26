r"""
coupling.py
"""
import torch
import torch.nn as nn

from reportengine import collect

ACTIVATION_LAYERS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    None: nn.Identity,
}
SYMMETRIC_ACTIVATIONS = ("tanh", None)

class Sequential(nn.Sequential):
    """Modify the nn.Sequential class so that it takes an input vector *and* a
    value for the current logarithm of the model density, returning an output
    vector and the updated log density."""

    def forward(self, x_input, log_density, *args):
        for module in self:
            x_input, log_density = module(x_input, log_density, *args)
        return x_input, log_density


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

    Methods
    -------
    forward:
        The forward pass of the network, mapping a batch of input vectors
        with 'size_in' nodes to a batch of output vectors of 'size_out'
        nodes.
    """

    def __init__(
        self,
        size_in: int,
        size_out: int,
        hidden_shape: list,
        activation: (str, None),
        final_activation: (str, None) = None,
        symmetric: bool = False,
    ):
        super(NeuralNetwork, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.hidden_shape = hidden_shape

        self.activation_func = ACTIVATION_LAYERS[activation]
        self.final_activation_func = ACTIVATION_LAYERS[final_activation]

        if symmetric:
            self.bias = False
            # TODO: check this in config
            #assert activation in SYMMETRIC_ACTIVATIONS
            #assert final_activation in SYMMETRIC_ACTIVATIONS
        else:
            self.bias = True

        # nn.Sequential object containing the network layers
        self.network = self._construct_network()

    def _block(self, f_in, f_out, activation_func):
        """Constructs a single 'dense block' which maps 'f_in' inputs to
        'f_out' output features. Returns a list with two elements:
            - Fully connected layer
            - Activation function
        """
        return [
            nn.Linear(f_in, f_out, bias=self.bias),
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
        """Forward pass of the network.
        
        Takes a tensor of shape (n_batch, size_in) and returns a new tensor of
        shape (n_batch, size_out)
        """
        return self.network(x)


_normalising_flow = collect("model_action", ("model_spec",))


def normalising_flow(_normalising_flow, i_mixture=1):
    """Return a callable model which is a normalising flow constructed via
    function composition."""
    return _normalising_flow[0]
