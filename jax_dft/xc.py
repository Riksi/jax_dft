from jax import tree_util
import jax.numpy as jnp

from jax_dft import constants


@tree_util.Partial
def exponential_coulomb_uniform_exchange_density(
    density,
    amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
    kappa=constants.EXPONENTIAL_COULOMB_KAPPA,
    epsilon=1e-15
):
    """Exchange energy density for uniform gas with exponential Coulomb

    Args:
        density:
        amplitude:
        kappa:
        epsilon:

    Returns:

    """
    y = jnp.pi * density / kappa
    return jnp.where(
        y > epsilon,
        jnp.log(y**2 + 1) / y - 2 * jnp.arctan(y),
        -y + y ** 3 / 6
    ) * amplitude / (2 * jnp.pi)


@tree_util.Partial
def exponential_coulumb_uniform_correlation_density(
    density,
    amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
    kappa=constants.EXPONENTIAL_COULOMB_KAPPA
):
    """

    Args:
        density:
        amplitude:
        kappa:

    Returns:

    """
    y = jnp.pi * density / kappa

    alpha = 2
    beta = -1.00077
    gamma = 6.26099
    delta = -11.9041
    eta = 9.62614
    sigma = -1.48334
    nu = 1.

    # Derivative of sqrt(x) not defined at x=0 so replace this
    # with approximation of -(amplitude/pi)*(y/alpha)
    finite_y = jnp.where(y == 0., 1., y)
    out = -finite_y / (
                gamma * finite_y  # 1
                + delta * jnp.sqrt(finite_y ** 3)  # 3/2
                + beta * jnp.sqrt(finite_y)  # 1/2
                + sigma * jnp.sqrt(finite_y ** 5)  # 5/2
                + eta * finite_y ** 2  # 2
                + nu * jnp.pi * kappa ** 2 / amplitude * finite_y ** 3  # 3
                + alpha  # 0
    ) * amplitude / jnp.pi
    return jnp.where(y == 0., -amplitude / jnp.pi * y / alpha, out)


@tree_util.Partial
def lda_xc_energy_density(
    density
):
    return (
        exponential_coulumb_uniform_correlation_density(
            density
        )
        + exponential_coulomb_uniform_exchange_density(
            density
        )
    )


def get_lda_xc_energy_density_fn():
    def lda_xc_energy_density_fn(density, params):
        del params
        return lda_xc_energy_density(density)
    return lda_xc_energy_density_fn