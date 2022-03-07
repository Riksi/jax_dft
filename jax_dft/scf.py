import functools
import typing
from typing import Optional, Union

import jax
from jax import tree_util
import jax.numpy as jnp
from jax_dft import utils

ArrayLike = Union[float, bool, jnp.ndarray]


def discrete_laplacian(num_grids):
    return (
        jnp.diag(-2.5 * jnp.ones(num_grids))
        + jnp.diag(4. / 3 * jnp.ones(num_grids - 1), k=1)
        + jnp.diag(4. / 3 * jnp.ones(num_grids - 1), k=-1)
        + jnp.diag(-1. / 12 * jnp.ones(num_grids - 2), k=2)
        + jnp.diag(-1. / 12 * jnp.ones(num_grids - 2), k=-2)
    )


@jax.jit
def get_kinetic_matrix(grids):
    dx = utils.get_dx(grids)
    return -0.5 * discrete_laplacian(grids.size) / (dx * dx)


@functools.partial(jax.jit, static_argnums=(0,))
def _wavefunctions_to_density(num_electrons, wavefunctions, grids):
    """Converts wavefunctions to density.

    Each eigenstate has two states: spin up and spin down

    Args:
        num_electrons: Integer, number of electrons in system.
            The first num_electrons states are occupied
        wavefunctions: float numpy array of
        shape [num_eigen_states, num_electrons]
        grids: float numpy array of shape (num_grids,)

    Returns:
        float numpy array of shape (num_grids,)
    """
    # Reduce to only occupied states
    # [N, E]
    wavefunctions = wavefunctions[:num_electrons]
    # Normalise
    # - N^2 = 1 / integral(|Psi|^2 * dV) = 1 / integral(|Psi|^2 * dx * dy)
    # ~ 1 / sum(|Psi|^2 * dx * dx) = 1 / (sum(|Psi|^2) * dx)
    # [N, E]
    wavefunctions = wavefunctions / jnp.sqrt(
        jnp.sum(wavefunctions ** 2, axis=1, keepdims=True)
        * utils.get_dx(grids)
    )
    # [w1, w2, ..., wN] -> [w1, w1, w2, w2, ..., wN, wN]
    intensities = jnp.repeat(wavefunctions ** 2, repeats=2, axis=0)
    # Then you sum [w1, w1, w2, w2, ..., w_{N/2}, w_{N/2}] if n even
    # or [w1, w1, w2, w2, ..., w_{(N+1)/2}] if n odd
    # That is because you need to consider two spins per state now
    # but for normalisation you normalise across the positions.
    return jnp.sum(intensities[:num_electrons], axis=0)


def wavefunctions_to_density(num_electrons, wavefunctions, grids):
    """Converts wavefunctions to density.

    Each eigenstate has two states: spin up and spin down

    Args:
        num_electrons: Integer, number of electrons in system. The first num_electrons states are occupied
        wavefunctions: float numpy array of
        shape [num_eigen_states, num_electrons]
        grids: float numpy array of shape (num_grids,)

    Returns:
        float numpy array of shape (num_grids,)
    """
    return _wavefunctions_to_density(num_electrons, wavefunctions, grids)


def get_total_eigen_energies(num_electrons, eigen_energies):
    """Gets total energies of first num_electron states

    Args:
        num_electrons:
        eigen_energies: [num_eigen_states]

    Returns:
    """
    return jnp.sum(
        jnp.repeat(eigen_energies, repeats=2)[:num_electrons]
    )


def get_gap(num_electrons, eigen_energies):
    """Gets the HOMO-LUMO gap

    HOMO: highest occupied molecular orbital
    LUMO: lowest unoccupied molecular orbital

    Args:
        num_electrons:
        eigen_energies:

    Returns:

    """

    double_occupied_eigen_energies = jnp.repeat(eigen_energies, repeats=2)
    lumo = double_occupied_eigen_energies[num_electrons]
    homo = double_occupied_eigen_energies[num_electrons - 1]
    return lumo - homo


@functools.partial(jax.jit, static_argnums=(1,))
def _solve_interacting_system(external_potential, num_electrons, grids):
    """Solves interacting system
    """
    eigen_energies, wavefunctions_transpose = jnp.linalg.eigh(
        # Hamiltonian matrix
        get_kinetic_matrix(grids) + jnp.diag(external_potential)
    )
    densities = wavefunctions_to_density(num_electrons,
        jnp.transpose(wavefunctions_transpose), grids)
    total_eigen_energies = get_total_eigen_energies(
        num_electrons=num_electrons,
        eigen_energies=eigen_energies)
    gap = get_gap(num_electrons, eigen_energies)
    return densities, total_eigen_energies, gap


def solve_interacting_system(external_potential, num_electrons, grids):
    """Solves noninteracting system

    Args:
        external_potential: [num_grids]
        num_electrons: int
        grids: [num_grids]

    Returns:
        density: [num_grids]
        total_eigen_energies: float
        gap: float

    """
    return _solve_interacting_system(
        external_potential, num_electrons, grids)


@functools.partial(jax.jit, static_argnums=(2,))
def _get_hartree_energy(density, grids, interaction_fn):
    """Gets the Hartee energy."""
    n1 = jnp.expand_dims(density, axis=0)
    n2 = jnp.expand_dims(density, axis=1)
    r1 = jnp.expand_dims(grids, axis=0)
    r2 = jnp.expand_dims(grids, axis=1)
    return 0.5 * jnp.sum(
        n1 * n2 * interaction_fn(r1 - r2)
    ) * utils.get_dx(grids) ** 2


def get_hartree_energy(density, grids, interaction_fn):
    """Gets the Hartee energy.

    U[n] = 0.5 \int dx \int dx' n(x) n(x') / \sqrt{(x - x')^2 + 1}

    Args:
        density:
        grids:
        interaction_fn: takes as input displacements and returns output
            float numpy array of the same shape

    Returns:

    """
    return _get_hartree_energy(density, grids, interaction_fn)


@functools.partial(jax.jit, static_argnums=(2,))
def _get_hartree_potential(density, grids, interaction_fn):
    """Gets the Hartee potential energy."""
    n1 = jnp.expand_dims(density, axis=0)  # [[n11, ..., n1N]]
    r1 = jnp.expand_dims(grids, axis=0)  # [[r11, ..., r1N]]
    r2 = jnp.expand_dims(grids, axis=1)  # [[r21], ..., [r2N]]
    # sum the elements of each row
    # so that result[i] = 0.5 * sum_z(n1z * f(r1z - r2i))
    # [[n11 * f(r11 - r21), ... n1N * f(r1N - r21)],
    #  ...
    #  [n11 * f(r11 - r2N), ... n1N * f(r1N - r2N)]]
    # numerically integrate for each r1 fixing r2
    return (#0.5 * \
           jnp.sum(
        n1 * interaction_fn(r1 - r2), axis=1
    ) * utils.get_dx(grids))


def get_hartree_potential(density, grids, interaction_fn):
    """Gets the Hartee potential.

    v_H(x) = \int dx' n(x') / \sqrt{(x - x')^2 + 1}

    Args:
        density:
        grids:
        interaction_fn: takes as input displacements and returns output
            float numpy array of the same shape

    Returns:

    """
    return _get_hartree_potential(density, grids, interaction_fn)


def get_external_potential_energy(density, external_potential, grids):
    """Gets external potential.

    Args:
        density:  [num_grids]
        external_potential: [num_grids]
        grids: [num_grids]

    Returns:

    """
    return jnp.dot(external_potential, density) * utils.get_dx(grids)


def get_xc_energy(density, xc_energy_density_function, grids):
    """Gets xc energy

    Args:
        density: [num_grids]
        xc_energy_density_function:
        grids: [num_grids]

    Returns:

    """

    return jnp.dot(xc_energy_density_function(density), density) * utils.get_dx(grids)


def get_xc_potential(density, xc_energy_density_function, grids):
    """Gets xc potential

    Finds this through automatic differentiation

    Args:
        density:
        xc_energy_density_function:
        grids:

    Returns:

    """

    return jax.grad(get_xc_energy)(density, xc_energy_density_function, grids) / utils.get_dx(grids)


def flip_and_average_fn(fn, locations, grids):
    """Flips and averages a function at the centre of the locations."""
    def output_fn(array):
        output_array = utils.flip_and_average(
            locations=locations, grids=grids, array=array
        )
        return utils.flip_and_average(
            locations=locations, grids=grids, array=fn(output_array)
        )
    return output_fn


class KohnShamState(typing.NamedTuple):
    """A named tuple containing the state of a Kohn-Sham iteration

    Attributes:
        density: float [num_grids]
        total_energy: float total energy of Kohn-Sham calculation
        locations: float [num_nuclei]
        nuclear_charges: float [num_nuclei]
        external_potential: float [num_grids]
        grids: float [num_grids]
        hartree_potential: float [num_grids]
        xc_potential: float [num_grids]
        xc_energy_density: float [num_grids]
        gap: Float, the Kohn-Sham gap
        converged: Boolean, whether the state is converged
    """

    density: jnp.ndarray
    total_energy: ArrayLike
    locations: jnp.ndarray
    nuclear_charges: jnp.ndarray
    external_potential: jnp.ndarray
    grids: jnp.ndarray
    num_electrons: ArrayLike
    hartree_potential: Optional[jnp.ndarray] = None
    xc_potential: Optional[jnp.ndarray] = None
    xc_energy_density: Optional[jnp.ndarray] = None
    gap: Optional[ArrayLike] = None
    converged: Optional[ArrayLike] = False


def kohn_sham_iteration(
    state,
    num_electrons,
    xc_energy_density_fn,
    interaction_fn,
    enforce_reflection_symmetry
):
    """One iteration of Kohn-Sham calculation

    xc_energy_density_fn must be wrapped in a jax.tree_util.Partial
    so that this function can take a callable. When the arguments of this
    callable changes, e.g. the parameters of the neural network,
    kohn_sham_iteration will not be recompiled

    Args:
        state:
        num_electrons:
        xc_energy_density_fn: [num_grids] -> [num_grids]
        interaction_fn:
        enforce_reflection_symmetry:

    Returns:

    """

    if enforce_reflection_symmetry:
        xc_energy_density_fn = flip_and_average_fn(
            xc_energy_density_fn,
            locations=state.locations,
            grids=state.grids
        )

    # v_H(r) = \int d^3r' w(r, r') n_0(r)
    hartree_potential = get_hartree_potential(
        density=state.density,
        grids=state.grids,
        interaction_fn=interaction_fn
    )

    # v_xc(r) = δE_xc/δn(r) at n = n_0
    xc_potential = get_xc_potential(
        density=state.density,
        xc_energy_density_function=xc_energy_density_fn,
        grids=state.grids
    )

    # v_s(r) = v_H(r) + v_ext(r) + v_xc(r)
    ks_potential = hartree_potential + state.external_potential + xc_potential
    xc_energy_density = xc_energy_density_fn(state.density)

    density, total_eigen_energies, gap = _solve_interacting_system(
        external_potential=ks_potential,
        num_electrons=num_electrons,
        grids=state.grids
    )

    # E[n] = T_s[n] + E_H[n] + E_ext[n] + E_xc[n]
    total_energy = (
        # T_s[n] - KS kinetic energy
        # T_s[n] = <T_s>
        # = <H_s> - <V_s>
        # = (sum of KS eigenstate energies)
        #     - (external potential energy due to KS potential)
        total_eigen_energies - get_external_potential_energy(
            density=density,
            external_potential=ks_potential,
            grids=state.grids
        )
        # E_H[n] - hartree energy
        + get_hartree_energy(
            density=density,
            interaction_fn=interaction_fn,
            grids=state.grids,
        )
        # E_ext[n] - external potential energy
        + get_external_potential_energy(
            density=density,
            external_potential=state.external_potential,
            grids=state.grids
        )
        # E_xc[n] = exchange correlation energy
        + get_xc_energy(
            density=density,
            xc_energy_density_function=xc_energy_density_fn,
            grids=state.grids
        )
    )

    if enforce_reflection_symmetry:
        density = utils.flip_and_average(
            locations=state.locations,
            grids=state.grids,
            array=density
        )

    return state._replace(
        density=density,
        hartree_potential=hartree_potential,
        xc_potential=xc_potential,
        xc_energy_density=xc_energy_density,
        total_energy=total_energy,
        gap=gap
    )


def kohn_sham(
    locations,
    nuclear_charges,
    num_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density=None,
    alpha=0.5,
    alpha_decay=0.9,
    enforce_reflection_symmetry=False,
    num_mixing_iterations=2,
    density_mse_converge_tolerance=-1.):
    """Runs Kohn-Sham to solve ground state of external potential

    Args:
        locations:
        nuclear_charges:
        num_electrons:
        num_iterations:
        grids:
        xc_energy_density_fn:
        interaction_fn:
        initial_density:
        alpha:
        alpha_decay:
        enforce_reflection_symmetry:
        num_mixing_iterations:
        density_mse_converge_tolerance:

    Returns:

    """
    external_potential = utils.get_atomic_chain_potential(
        locations=locations,
        grids=grids,
        nuclear_charges=nuclear_charges,
        interaction_fn=interaction_fn
    )

    if initial_density is None:
        # Use non-interacting solution from external potential as initial guess
        initial_density, _, _ = solve_interacting_system(
            external_potential=external_potential,
            num_electrons=num_electrons,
            grids=grids
        )
    # Create initial state.
    state = KohnShamState(
        density=initial_density,
        total_energy=jnp.inf,
        locations=locations,
        nuclear_charges=nuclear_charges,
        external_potential=external_potential,
        grids=grids,
        num_electrons=num_electrons,
    )
    states = []
    differences = None
    converged = False

    for _ in range(num_iterations):
        if converged:
            states.append(state)
            continue

        old_state = state
        state = kohn_sham_iteration(
            state=state,
            num_electrons=num_electrons,
            xc_energy_density_fn=xc_energy_density_fn,
            interaction_fn=interaction_fn,
            enforce_reflection_symmetry=enforce_reflection_symmetry
        )

        density_difference = state.density - old_state.density

        if differences is None:
            differences = jnp.array([density_difference])
        else:
            differences = jnp.vstack([differences, density_difference])

        if jnp.mean(jnp.square(differences)) < density_mse_converge_tolerance:
            converged = True

        state = state._replace(converged=converged)
        state = state._replace(
            density=old_state.density + alpha *
                jnp.mean(differences[-num_mixing_iterations:], axis=0)
        )
        states.append(state)
        alpha *= alpha_decay

    return tree_util.tree_multimap(lambda *x: jnp.stack(x), *states)


def get_final_state(state):
    return jax.tree_map(lambda x: x[-1], state)


def state_iterator(state):
    leaves, treedef = tree_util.tree_flatten(state)
    for elements in zip(*leaves):
        yield treedef.unflatten(elements)


def get_initial_density(states, method):
    if method == "exact":
        return states.density
    elif method == "noninteracting":
        solve = jax.vmap(
            solve_interacting_system,
            in_axes=(0, None, None)
        )
        return solve(
            states.external_potential,
            states.num_electrons[0],
            states.grids[0]
        )[0]
    else:
        raise ValueError(f"Unknown initialisation method {method}")







