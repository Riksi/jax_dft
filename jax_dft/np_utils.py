
from jax import tree_util
import numpy as np
from scipy import special

from jax_dft import constants
from jax_dft import utils


def flatten(params, dtype=np.float64):
    leaves, tree = tree_util.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    vec = np.concatenate([leaf.ravel() for leaf in leaves]).astype(dtype)
    return (tree, shapes), vec


def unflatten(spec, vec):
    tree, shapes = spec
    sizes = [int(np.prod(shape)) for shape in shapes]
    leaves_flat = np.split(vec, np.cumsum(sizes)[:-1])
    leaves = [np.reshape(leaf, shape) for leaf, shape in zip(leaves_flat, shapes)]
    return tree_util.tree_unflatten(tree, leaves)


def _get_exact_h_atom_density(displacements, dx, energy=-0.670):
    """

    Args:
        displacements: [num_nuclei, num_grids]
        dx: float
        energy: float

    Returns:

    """
    v = np.sqrt(-8 * energy / constants.EXPONENTIAL_COULOMB_KAPPA ** 2)
    # [num_nuclei, num_grids]
    z = (2 / constants.EXPONENTIAL_COULOMB_KAPPA) * np.sqrt(
        2 * constants.EXPONENTIAL_COULOMB_AMPLITUDE) * np.exp(
            -constants.EXPONENTIAL_COULOMB_KAPPA * np.abs(displacements) / 2
        )

    raw_exact_density = special.jv(v, z) ** 2

    # [num_nuclei, num_grids] / [num_nuclei, 1] -> [num_nuclei, num_grids]
    return raw_exact_density / (
            np.sum(raw_exact_density, axis=1, keepdims=True) * dx)


def spherical_superposition_density(grids, locations, nuclear_charges):
    """

    Args:
        grids: [num_grids]
        locations: [num_nuclei]
        nuclear_charges: [num_nuclei]

    Returns:
        Float numpy array with shape [num_grids]

    """
    displacements = np.expand_dims(
        np.array(grids), axis=0) - np.expand_dims(np.array(locations), axis=1)
    densities = _get_exact_h_atom_density(displacements, float(utils.get_dx(grids)))
    return np.dot(nuclear_charges, densities)



