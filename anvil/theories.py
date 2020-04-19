"""
distributions.py

Module containing classes corresponding to different base distributions.
"""
import torch
import torch.nn as nn


class PhiFourAction(nn.Module):
    """Extend the nn.Module class to return the phi^4 action given either
    a single state size (1, length * length) or a stack of N states
    (N, length * length). See Notes about action definition.

    The forward pass returns the corresponding log density (unnormalised) which
    is equal to -S

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    m_sq: float
        the value of the bare mass squared
    lam: float
        the value of the bare coupling

    Examples
    --------
    Consider the toy example of this class acting on a random state

    >>> geom = Geometry2D(2)
    >>> action = PhiFourAction(1, 1, geom)
    >>> state = torch.rand((1, 2*2))
    >>> action(state)
    tensor([[-2.3838]])
    >>> state = torch.rand((5, 2*2))
    >>> action(state)
    tensor([[-3.9087],
            [-2.2697],
            [-2.3940],
            [-2.3499],
            [-1.9730]])

    Notes
    -----
    that this is the action as defined in
    https://doi.org/10.1103/PhysRevD.100.034515 which might differ from the
    current version on the arxiv.

    """

    def __init__(self, m_sq, lam, geometry, use_arxiv_version=False):
        super(PhiFourAction, self).__init__()
        self.shift = geometry.get_shift()
        self.lam = lam
        self.m_sq = m_sq
        self.lattice_size = geometry.length ** 2
        if use_arxiv_version:
            self.version_factor = 2
        else:
            self.version_factor = 1

    def forward(self, phi_state: torch.Tensor) -> torch.Tensor:
        """Perform forward pass, returning -action for stack of states. Note
        here the minus sign since we want to return the log density of the
        corresponding unnormalised distribution

        see class Notes for details on definition of action.
        """
        action = (
            self.version_factor * (2 + 0.5 * self.m_sq) * phi_state ** 2  # phi^2 terms
            + self.lam * phi_state ** 4  # phi^4 term
            - self.version_factor
            * torch.sum(
                phi_state[:, self.shift] * phi_state.view(-1, 1, self.lattice_size),
                dim=1,
            )  # derivative
        ).sum(
            dim=1, keepdim=True  # sum across sites
        )
        return -action


class XYHamiltonian(nn.Module):
    """
    Extend the nn.Module class to return the Hamiltonian for the classical
    XY spin model (also known as the O(2) model), given a stack of polar
    angles with shape (sample_size, lattice_size).

    The spins are defined as having modulus 1, such that they take values
    on the unit circle.

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    beta: float
        the inverse temperature (coupling strength).
    """

    def __init__(self, beta, geometry):
        super().__init__()
        self.beta = beta
        self.lattice_size = geometry.length ** 2
        self.shift = geometry.get_shift()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute XY Hamiltonian from a stack of angles (not Euclidean field components)
        with shape (sample_size, lattice_size).
        """
        hamiltonian = -self.beta * torch.cos(
            state[:, self.shift] - state.view(-1, 1, self.lattice_size)
        ).sum(
            dim=1,
        ).sum(  # sum over two shift directions (+ve nearest neighbours)
            dim=1, keepdim=True
        )  # sum over lattice sites
        return -hamiltonian


class HeisenbergHamiltonian(nn.Module):
    """
    Extend the nn.Module class to return the Hamiltonian for the classical
    Heisenberg model (also known as the O(3) model), given a stack of polar
    angles with shape (sample_size, 2 * lattice_size).

    The spins are defined as having modulus 1, such that they take values
    on the unit 2-sphere, and can be parameterised by two angles using
    spherical polar coordinates (with the radial coordinate equal to one).

    Parameters
    ----------
    geometry:
        define the geometry of the lattice, including dimension, size and
        how the state is split into two parts
    beta: float
        the inverse temperature (coupling strength).
    """

    def __init__(self, beta, geometry):
        super().__init__()
        self.beta = beta
        self.lattice_size = geometry.length ** 2
        self.shift = geometry.get_shift()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        r"""
        Compute classical Heisenberg Hamiltonian from a stack of angles with shape
        (sample_size, 2 * volume).

        Also computes the logarithm of the 'volume element' for the probability
        distribution due to parameterisating the spin vectors using polar coordinates.
        
        The volume element for a configuration is a product over all lattice sites
        
            \prod_{n=1}^V sin(\theta_n)

        where \theta_n is the polar angle for the spin at site n.
        
        Notes
        -----
        Assumes that state.view(-1, lattice_size, 2) yields a tensor for which the
        two elements in the final dimension represent, respectively, the polar and
        azimuthal angles for the same lattice site.
        """
        polar = state[:, ::2]
        azimuth = state[:, 1::2]
        cos_polar = torch.cos(polar)
        sin_polar = torch.sin(polar)

        hamiltonian = -self.beta * (
            cos_polar[:, self.shift] * cos_polar.view(-1, 1, self.lattice_size)
            + sin_polar[:, self.shift]
            * sin_polar.view(-1, 1, self.lattice_size)
            * torch.cos(azimuth[:, self.shift] - azimuth.view(-1, 1, self.lattice_size))
        ).sum(
            dim=1,
        ).sum(  # sum over two shift directions (+ve nearest neighbours)
            dim=1, keepdim=True
        )  # sum over lattice sites

        log_volume_element = torch.log(sin_polar).sum(dim=1, keepdim=True)
        
        return log_volume_element - hamiltonian


def phi_four_action(m_sq, lam, geometry, use_arxiv_version):
    """returns instance of PhiFourAction"""
    return PhiFourAction(
        m_sq, lam, geometry=geometry, use_arxiv_version=use_arxiv_version
    )


def xy_hamiltonian(beta, geometry):
    return XYHamiltonian(beta, geometry)


def heisenberg_hamiltonian(beta, geometry):
    return HeisenbergHamiltonian(beta, geometry)
