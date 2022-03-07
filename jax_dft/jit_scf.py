"""Functions for self-consistent field calculation simplified for jit."""

import functools

import jax
import jax.numpy as jnp

from jax_dft import scf
from jax_dft import utils


def _flip_and_average_on_centre(array):
    return (array + jnp.flip(array)) / 2


def _flip_and_average_on_centre_fn(fn):
    def average_fn(array):
        return _flip_and_average_on_centre(
            fn(_flip_and_average_on_centre(array))
        )
    return average_fn


def _connection_weights(num_iterations, num_mixing_iterations):
    mask = jnp.triu(
        jnp.tril(
            jnp.ones((num_iterations, num_iterations))
        ), k=-num_mixing_iterations + 1
    )
    return mask / jnp.sum(mask, axis=1, keepdims=True)


@functools.partial(jax.jit, static_argnums=(3, 5, 6))
def _kohn_sham_iteration(
    density,
    external_potential,
    grids,
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
    # Seems that for jit use can't use KohnShamState

    if enforce_reflection_symmetry:
        xc_energy_density_fn = _flip_and_average_on_centre_fn(
            xc_energy_density_fn,
        )

    # v_H(r) = \int d^3r' w(r, r') n_0(r)
    hartree_potential = scf.get_hartree_potential(
        density=density,
        grids=grids,
        interaction_fn=interaction_fn
    )

    # v_xc(r) = δE_xc/δn(r) at n = n_0
    xc_potential = scf.get_xc_potential(
        density=density,
        xc_energy_density_function=xc_energy_density_fn,
        grids=grids
    )

    # v_s(r) = v_H(r) + v_ext(r) + v_xc(r)
    ks_potential = hartree_potential + external_potential + xc_potential
    xc_energy_density = xc_energy_density_fn(density)

    density, total_eigen_energies, gap = scf._solve_interacting_system(
        external_potential=ks_potential,
        num_electrons=num_electrons,
        grids=grids
    )

    # E[n] = T_s[n] + E_H[n] + E_ext[n] + E_xc[n]
    total_energy = (
        # T_s[n] - KS kinetic energy
        # T_s[n] = <T_s>
        # = <H_s> - <V_s>
        # = (sum of KS eigenstate energies)
        #     - (external potential energy due to KS potential)
        total_eigen_energies - scf.get_external_potential_energy(
            density=density,
            external_potential=ks_potential,
            grids=grids
        )
        # E_H[n] - hartree energy
        + scf.get_hartree_energy(
            density=density,
            interaction_fn=interaction_fn,
            grids=grids,
        )
        # E_ext[n] - external potential energy
        + scf.get_external_potential_energy(
            density=density,
            external_potential=external_potential,
            grids=grids
        )
        # E_xc[n] = exchange correlation energy
        + scf.get_xc_energy(
            density=density,
            xc_energy_density_function=xc_energy_density_fn,
            grids=grids
        )
    )

    if enforce_reflection_symmetry:
        density = _flip_and_average_on_centre(density)

    return (
        density,
        total_energy,
        hartree_potential,
        xc_potential,
        xc_energy_density,
        gap
    )


def kohn_sham_iteration(
    state,
    num_electrons,
    xc_energy_density_fn,
    interaction_fn,
    enforce_reflection_symmetry
):
    (
        density,
        total_energy,
        hartree_potential,
        xc_potential,
        xc_energy_density,
        gap) = _kohn_sham_iteration(
            state.density,
            state.external_potential,
            state.grids,
            num_electrons,
            xc_energy_density_fn,
            interaction_fn,
            enforce_reflection_symmetry)
    return state._replace(
        density=density,
        total_energy=total_energy,
        hartree_potential=hartree_potential,
        xc_potential=xc_potential,
        xc_energy_density=xc_energy_density,
        gap=gap)


@functools.partial(jax.jit, static_argnums=(2, 3, 6, 8, 9, 10, 11, 12, 13))
def _kohn_sham(
    locations,
    nuclear_charges,
    num_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density,
    alpha,
    alpha_decay,
    enforce_reflection_symmetry,
    num_mixing_iterations,
    density_mse_converge_tolerance,
    stop_gradient_step):
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

    num_grids = grids.shape[0]
    weights = _connection_weights(num_iterations, num_mixing_iterations)

    def _converged_kohn_sham_iteration(old_state_differences):
        old_state, differences = old_state_differences
        return old_state._replace(converged=True), differences

    def _unconverged_kohn_sham_iteration(idx_old_state_alpha_differences):
        idx, old_state, alpha, differences = idx_old_state_alpha_differences
        state = kohn_sham_iteration(
            state=old_state,
            num_electrons=num_electrons,
            xc_energy_density_fn=xc_energy_density_fn,
            interaction_fn=interaction_fn,
            enforce_reflection_symmetry=enforce_reflection_symmetry
        )
        differences.at[idx].set(state.density - old_state.density)
        # Density mixing
        state = state._replace(
            density=old_state.density + alpha *
                    jnp.dot(weights[idx], differences)
        )
        return state, differences

    def _single_kohn_sham_iteration(carry, inputs):
        del inputs
        idx, old_state, alpha, converged, differences = carry
        state, differences = jax.lax.cond(
            converged,
            true_operand=(old_state, differences),
            true_fun=_converged_kohn_sham_iteration,
            false_operand=(idx, old_state, alpha, differences),
            false_fun=_unconverged_kohn_sham_iteration
        )

        converged = jnp.mean(
            jnp.square(differences)) < density_mse_converge_tolerance
        state = jax.lax.cond(
            idx <= stop_gradient_step,
            true_fn=jax.lax.stop_gradient,
            false_fn=lambda x: x,
            operand=state
        )
        return (idx + 1, state, alpha * alpha_decay, converged, differences), state

    state = scf.KohnShamState(
            density=initial_density,
            total_energy=jnp.inf,
            locations=locations,
            nuclear_charges=nuclear_charges,
            grids=grids,
            external_potential=utils.get_atomic_chain_potential(
                grids=grids,
                locations=locations,
                nuclear_charges=nuclear_charges,
                interaction_fn=interaction_fn
            ),
            num_electrons=num_electrons,
            # dummy fields so all states have same structure for scan
            hartree_potential=jnp.zeros_like(grids),
            xc_potential=jnp.zeros_like(grids),
            xc_energy_density=jnp.zeros_like(grids),
            gap=0.,
            converged=False
        )

    differences = jnp.zeros((num_iterations, num_grids))

    _, states = jax.lax.scan(
        _single_kohn_sham_iteration,
        init=(0, state, alpha, state.converged, differences),
        xs=jnp.arange(num_iterations)
    )

    return states




def kohn_sham(
    locations,
    nuclear_charges,
    num_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density,
    alpha=0.5,
    alpha_decay=0.9,
    enforce_reflection_symmetry=False,
    num_mixing_iterations=2,
    density_mse_converge_tolerance=-1.,
    stop_gradient_step=-1
):
    """
    Changes to make it work with jit are
    * No default value for initial density
    * No convergence criteria and early stopping
    * Reflection symmetry flips density only about centre not at locations

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
        stop_gradient_step:

    Returns:

    """

    return _kohn_sham(
        locations,
        nuclear_charges,
        num_electrons,
        num_iterations,
        grids,
        xc_energy_density_fn,
        interaction_fn,
        initial_density,
        alpha,
        alpha_decay,
        enforce_reflection_symmetry,
        num_mixing_iterations,
        density_mse_converge_tolerance,
        stop_gradient_step)



