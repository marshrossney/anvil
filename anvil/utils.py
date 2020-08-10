import numpy as np
import multiprocessing as mp
from itertools import islice
from functools import wraps
from math import ceil
import torch

USE_MULTIPROCESSING = True


class Multiprocessing:
    """Class which implements multiprocessing of a function given a number
    inputs for that function.

    Parameters
    ----------
    func: function/method
        the function to be executed multiple times
    generator: function/method
        something which, when called, returns a generator object that contains
        the parameters for the function.

    Notes
    -----
    Does not rely on multiprocessing.Pool since that does not work with
    instance methods without considerable extra effort (it cannot pickle them).
    """

    def __init__(self, func, generator):
        self.func = func
        self.generator = generator

        self.n_iters = sum(1 for _ in generator())
        self.n_cores = mp.cpu_count()
        if not USE_MULTIPROCESSING:
            self.n_cores = 1
        self.max_chunk = ceil(self.n_iters / self.n_cores)

    def target(self, k, output_dict):
        """Function to be executed for each process."""
        generator_k = islice(
            self.generator(),
            k * self.max_chunk,
            min((k + 1) * self.max_chunk, self.n_iters),
        )
        i_glob = k * self.max_chunk  # global index
        for i, args in enumerate(generator_k):
            output_dict[i_glob + i] = self.func(args)
        return

    def __call__(self):
        """Returns a dictionary containing the function outputs for each
        set of parameters taken from the generator. The dictionary keys are
        integers which label the order of parameter sets in the generator."""
        manager = mp.Manager()
        output_dict = manager.dict()

        procs = []
        for k in range(self.n_cores):
            p = mp.Process(target=self.target, args=(k, output_dict,),)
            procs.append(p)
            p.start()

        # Kill the zombies
        for p in procs:
            p.join()

        return output_dict


def bootstrap_sample(data, bootstrap_sample_size):
    state = np.random.RandomState()
    *dims, data_size = data.shape

    sample = []
    for j in range(bootstrap_sample_size):
        boot_index = state.randint(0, data_size, size=data_size)
        sample.append(data[..., boot_index])

    return np.stack(sample, axis=-2)


def spher_to_eucl(coords):
    """Converts a set (N-1) angles to a set of N-component euclidean unit vectors.

    # TODO
    The order of the (N-1) angles [\phi^0, ..., \phi^{N-1}] is taken to match some
    convention.

    Parameters
    ----------
    coords: numpy.ndarray
        The spherical coordinates (angles). The (N-1) angles are expected on the 1st
        dimension. Dimension (lattice.volume, (N-1), *).

    Returns
    -------
    out: numpy.ndarray
        The Euclidean representation of the angles, dimension (lattice.volume, N, *).

    Notes
    -----
    See REF
    """
    output_shape = list(coords.shape)
    output_shape[1] += 1

    output = np.ones(output_shape)
    output[:, :-1] = np.cos(coords)
    output[:, 1:] *= np.cumprod(np.sin(coords), axis=1)
    return output


class unit_norm:
    class UnitNormError(Exception):
        pass

    def __init__(self, dim=1, atol=1e-6):
        self._dim = dim
        self._atol = atol

    def __call__(self, setter):
        @wraps(setter)
        def wrapper(instance, array_in):
            if not np.allclose(
                np.linalg.norm(array_in, axis=self._dim), 1, atol=self._atol
            ):
                raise self.UnitNormError(
                    f"Array contains elements with a norm along dimension {self.dim} that deviates from unity by more than {self.atol}."
                )

            setter(instance, array_in)

def get_num_parameters(model):
    """Return the number of trainable parameters in a model.

    Taken from github.com/bayesiains/nflows
    """
    num = 0
    for parameter in model.parameters():
        num += torch.numel(parameter)
    return num
