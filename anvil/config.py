"""
config.py

Module to parse runcards
"""
import logging

from reportengine.report import Config
from reportengine.configparser import ConfigError, element_of

from anvil.core import PhiFourAction, NVectorAction, TrainingOutput
from anvil.models import RealNVP, NonCompactProjection
from anvil.geometry import Geometry2D

log = logging.getLogger(__name__)

from anvil.utils import SphericalUniformDist


class ConfigParser(Config):
    """Extend the reportengine Config class for <project name> specific
    objects
    """

    def parse_lattice_length(self, length: int):
        return length

    def parse_lattice_dimension(self, dim: int):
        """Parse lattice dimension from runcard"""
        if dim != 2:
            raise ConfigError("Currently only 2 dimensions is supported")
        return dim

    def produce_lattice_size(self, lattice_length, lattice_dimension):
        """returns the total number of nodes on lattice"""
        return pow(lattice_length, lattice_dimension)

    def parse_n_affine(self, n: int):
        return n

    def parse_epochs(self, epochs: int):
        return epochs

    @element_of("cp_ids")
    def parse_cp_id(self, cp: (int, type(None))):
        return cp

    @element_of("training_outputs")
    def parse_training_output(self, path: str):
        return TrainingOutput(path)

    @element_of("checkpoints")
    def produce_checkpoint(self, cp_id=None, training_output=None):
        if cp_id is None:
            return None
        if cp_id == -1:
            return training_output.final_checkpoint()
        if cp_id not in training_output.cp_ids:
            raise ConfigError(f"Checkpoint {cp_id} not found in {training_output.path}")
        # get index from training_output class
        return training_output.checkpoints[training_output.cp_ids.index(cp_id)]

    def parse_hidden_nodes(self, hid_spec):
        return hid_spec

    def produce_network_kwargs(self, hidden_nodes):
        """Returns a dictionary that is the necessary kwargs for the NVP class
        This means in the future if we change the class to have more flexibility
        with regard to network spec then we can use this function to bridge
        backwards compatibility"""
        hidden_nodes = tuple(hidden_nodes)
        return dict(affine_hidden_shape=hidden_nodes)

    """
    def produce_model(self, lattice_size, n_affine, network_kwargs):
        model = RealNVP(n_affine=n_affine, size_in=lattice_size, **network_kwargs)
        return model
    """

    def produce_model(self, generator, n_affine, network_kwargs):
        model = NonCompactProjection(
            generator=generator, n_affine=n_affine, **network_kwargs,
        )
        return model

    def parse_optimiser_input(self, optim):
        raise NotImplementedError

    def parse_m_sq(self, m: (float, int)):
        return m

    def parse_lam(self, lam: (float, int)):
        return lam

    def parse_use_arxiv_version(self, do_use: bool):
        return do_use

    def parse_beta(self, beta):
        return beta

    def parse_n_coords(self, n: int):
        return n

    def produce_geometry(self, lattice_length):
        return Geometry2D(lattice_length)

    def produce_generator(self, n_coords, lattice_length):
        return SphericalUniformDist(dimension=n_coords, lattice_length=lattice_length)

    """
    def produce_action(self, m_sq, lam, geometry, use_arxiv_version):
        return PhiFourAction(
            m_sq, lam, geometry=geometry, use_arxiv_version=use_arxiv_version
        )
    """

    def produce_action(self, n_coords, beta, geometry, shift_action):
        return NVectorAction(n_coords, beta, geometry, shift_action)

    def parse_target_length(self, targ: int):
        return targ

    def parse_thermalisation(self, therm: (int, type(None))):
        if therm is None:
            log.warning("Not Performing thermalisation")
            return therm
        if therm < 1:
            raise ConfigError(
                "thermalisation must be greater than or equal to 1 or be None"
            )
        return therm

    def parse_sample_interval(self, interval: (int, type(None))):
        if interval is None:
            return interval
        if interval < 1:
            raise ConfigError("sample_interval must be greater than or equal to 1")
        log.warning(f"Using user specified sample_interval: {interval}")
        return interval

    def parse_bootstrap_n_samples(self, n_samples: int):
        if n_samples < 2:
            raise ConfigError("bootstrap_n_samples must be greater than 1")
        log.warning(f"Using user specified bootstrap_n_samples: {n_samples}")
        return n_samples

    @element_of("windows")
    def parse_window(self, window: float):
        if window < 0:
            raise ConfigError("window must be positive")
        log.warning(f"Using user specified window 'S': {window}")
        return window

    def produce_training_context(self, training_output):
        """Given a training output produce the context of that training"""
        with self.set_context(ns=self._curr_ns.new_child(training_output.as_input())):
            _, geometry = self.parse_from_(None, "geometry", write=False)
            _, model = self.parse_from_(None, "model", write=False)
            _, action = self.parse_from_(None, "action", write=False)
            _, cps = self.parse_from_(None, "checkpoints", write=False)

        return dict(geometry=geometry, model=model, action=action, checkpoints=cps)

    def produce_training_geometry(self, training_output):
        with self.set_context(ns=self._curr_ns.new_child(training_output.as_input())):
            _, geometry = self.parse_from_(None, "geometry", write=False)
        return geometry
