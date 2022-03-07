import jax
import jax.numpy as jnp

from jax_dft import constants


def shift(array, offset):
    shifted = array[slice(offset, None) if offset >= 0 else slice(None, offset)]
    return jnp.pad(
        shifted,
        # if left is trimmed then offset >=0 so you get (0, offset)
        # if right is trimmed then offset < 0 so you get (-offset, 0)
        pad_width=(-min(offset, 0), max(offset, 0)),
        mode='constant',
        constant_values=0
    )


def get_dx(grids):
    if grids.ndim != 1:
        raise ValueError(f"grids.ndim is expected to be 1 but got {grids.ndim}")
    return (jnp.amax(grids) - jnp.amin(grids)) / (grids.size - 1)


def gaussian(grids, centre, sigma=1.):
    return 1. / jnp.sqrt(2 * jnp.pi) * jnp.exp(
        -0.5 * ((grids - centre) / sigma) ** 2
    ) / sigma


def soft_coulomb(displacements, soften_factor=constants.SOFT_COULOMB_SOFTEN_FACTOR):
    """Soft Coulomb interaction

    Args:
        displacements:
        soften_factor:

    Returns:

    """
    return 1 / jnp.sqrt(displacements ** 2 + soften_factor)


def exponential_coulomb(
    displacements,
    amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
    kappa=constants.EXPONENTIAL_COULOMB_KAPPA
):
    return amplitude * jnp.exp(kappa * -jnp.abs(displacements))


def get_atomic_chain_potential(
    grids,
    locations,
    nuclear_charges,
    interaction_fn
):
    """

    Args:
        grids: (num_grids,)
        locations: (num_locations,) - locations of nuclei
        nuclear_charges: (num_locations,) - nuclear charges
        interaction_fn:

    Returns:

    """
    if grids.ndim != 1:
        raise ValueError(f"grids.ndim is expected to be 1 but got {grids.ndim}")

    if locations.ndim != 1:
        raise ValueError(f"locations.ndim is expected to be 1 but got {locations.ndim}")

    if nuclear_charges.ndim != 1:
        raise ValueError(f"nuclear_charges.ndim is expected to be 1 but got {nuclear_charges.ndim}")

    # All the pairwise displacements
    # [1, G] - [L, 1] -> [L, G]
    displacements = jnp.expand_dims(
        grids, axis=0
    ) - jnp.expand_dims(
        locations, axis=1
    )

    # [L], [L, G] -> [G]
    # result[i] = sum_a(-Za * f(xi - Xa))
    # I think atomic units are used so that
    # e = 1 which means electron charge -e = -1
    return jnp.dot(
        -nuclear_charges, interaction_fn(displacements)
    )


def get_nuclear_interaction_energy(
    locations, nuclear_charges, interaction_fn
):
    """Gets nuclear interaction energy for atomic chain
    \sum_{i<j}\text{interaction_fn}(x_i - x_j) * \text{charge}_i * \text{charge}_j

    Args:
        locations:
        nuclear_charges:
        interaction_fn:

    Returns:

    """

    locations = jnp.array(locations)
    nuclear_charges = jnp.array(nuclear_charges)

    if locations.ndim != 1:
        raise ValueError(f"locations.ndim is expected to be 1 but got {locations.ndim}")

    if nuclear_charges.ndim != 1:
        raise ValueError(f"nuclear_charges.ndim is expected to be 1 but got {nuclear_charges.ndim}")

    indices_0, indices_1 = jnp.triu_indices(locations.size, k=1)

    charges_product = nuclear_charges[indices_0] * nuclear_charges[indices_1]
    return jnp.sum(
        charges_product * interaction_fn(
            locations[indices_0] - locations[indices_1]))


def get_nuclear_interaction_energy_batch(
    locations, nuclear_charges, interaction_fn
):
    return jax.vmap(
        get_nuclear_interaction_energy,
        in_axes=(0, 0, None)
    )(locations, nuclear_charges, interaction_fn)


def _float_value_in_array(array, value, atol=1e-7):
    return any(jnp.abs(array - value) <= atol)


def location_centre_at_grids_centre_point(locations, grids):
    """Checks whether the centre of the location is at the centre of the grids

    Args:
        grids:
        locations:

    Returns:

    """
    num_grids = grids.shape[0]
    return bool(
        num_grids % 2
        and jnp.abs(jnp.mean(locations) - grids[num_grids//2]) < 1e-8)


def flip_and_average(locations, grids, array):
    for location in locations:
        if not _float_value_in_array(grids, location):
            raise ValueError(f"Location {location:4.2f} is not on the grids")
    centre = jnp.mean(locations)
    if _float_value_in_array(centre, grids):
        centre_index = jnp.argmin(jnp.abs(grids - centre))
        left_index = centre_index
        right_index = centre_index
    else:
        abs_distance_to_centre = jnp.abs(grids - centre)
        left_index = jnp.argmin(
            jnp.where(grids < centre, abs_distance_to_centre, jnp.inf)
        )
        right_index = jnp.argmin(
            jnp.where(grids > centre, abs_distance_to_centre, jnp.inf)
        )

    # grid = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # centre = 4.2
    # -> [0, 1, 2, 3, 4], [5, 6, 7, 8]
    # -> L=4, R=5, r=min(L=4, 9-R-1=3) = 3
    # -> range_slice=(-3+L, 3+R+1) = (1, 9)
    # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # [-, 8, 7, 6, 5, 4, 3, 2, 1]

    # centre = 3.2
    # -> [0, 1, 2, 3], [4, 5, 6, 7, 8]
    # -> L=3, R=4, r=min(L=3,9-R-1=4) = 3
    # -> range_slice=(-3+L, 3+R+1) = (0, 8)
    # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # [7, 6, 5, 4, 3, 2, 1, 0, -]

    # Note that the pairs of arrays above are of the indices
    # not the values themselves
    radius = min(left_index, len(grids) - right_index - 1)
    range_slice = slice(left_index - radius, right_index + radius + 1)
    array_to_flip = array[range_slice]
    return array.at[range_slice].set(
        (array_to_flip + jnp.flip(array_to_flip)) / 2
    )


def compute_distances_between_nuclei(locations, nuclei_indices):
    """Computes the distances between nuclei.

    Args:
        locations: [batch_size, num_nuclei]
        nuclei_indices: Tuple of two integers, the indices of nuclei
            to compute distances

    Returns:

    """
    if locations.ndim != 2:
        raise ValueError(f"locations.ndim is expected to be 2 but got {locations.ndim}")

    size = len(nuclei_indices)
    if size != 2:
        raise ValueError(f"size of nuclei_indices is expected to be 2 but got {size}")

    return jnp.abs(
        locations[:, nuclei_indices[0]] - locations[:, nuclei_indices[1]]
    )
