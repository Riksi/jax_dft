from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np

from jax_dft import constants
from jax_dft import xc

config.update('jax_enable_x64', True)

class XcTest(parameterized.TestCase):

    def setUp(self):
        super(XcTest, self).setUp()
        self.amplitude = constants.EXPONENTIAL_COULOMB_AMPLITUDE
        self.kappa = constants.EXPONENTIAL_COULOMB_KAPPA

    def test_exponential_coulomb_uniform_exchange_density(self):
        # Does not work for 1e-5 unless you do
        # `config.update('jax_enable_x64', True)`
        np.testing.assert_allclose(
            xc.exponential_coulomb_uniform_exchange_density(
                density=jnp.array([1e-15, 1e-10, 1e-5, 1., 2., 3., 20.])
            ),
            [0., 0., -0.000013, -0.3983580525, -0.4512822906, -0.4732600516, -0.521973703],
            atol=1e-6
        )

    def test_exponential_coulomb_uniform_exchange_density_low_density_limit(self):
        density = jnp.linspace(0, 0.01, 5)
        y = jnp.pi * density / self.kappa
        np.testing.assert_allclose(
            xc.exponential_coulomb_uniform_exchange_density(density=density),
            self.amplitude / (2 * jnp.pi) * (-y + y ** 3 / 6),
            atol=1e-6
        )

    def test_exponential_coulomb_uniform_exchange_density_high_density_limit(self):
        limit_value = -self.amplitude / 2
        np.testing.assert_allclose(
            xc.exponential_coulomb_uniform_exchange_density(
                density=jnp.array([1000., 10000.])
            ),
            [limit_value, limit_value],
            atol=1e-3
        )

    def test_exponential_coulomb_uniform_exchange_density_gradient(self):
        grad_fn = jax.vmap(
            jax.grad(xc.exponential_coulomb_uniform_exchange_density),
            in_axes=(0,)
        )
        density = jnp.linspace(0, 3, 11)
        y = jnp.pi * density / self.kappa
        dEx_dn = -self.amplitude / (2 * self.kappa) * jnp.log(1 + y ** 2) / y ** 2

        np.testing.assert_allclose(
            grad_fn(density),
            dEx_dn
        )

    def test_exponential_coulomb_uniform_correlation_density_low_density_limit(self):
        alpha = 2
        beta = -1.00077
        density = jnp.linspace(0, 0.005, 5)
        y = jnp.pi * density / self.kappa

        np.testing.assert_allclose(
            xc.exponential_coulumb_uniform_correlation_density(density=density),
            # Approximate denominator as the sum of constant term
            # and term with the largest power of y which is sqrt(y)
            # so you have 1 / (alpha + beta * sqrt(y))
            # ~ 1/alpha * (1 - beta / alpha * sqrt(y))
            self.amplitude / (jnp.pi * alpha) * (-y * (1 - beta / alpha * jnp.sqrt(y))),
            atol=1e-3
        )

    def test_exponential_coulomb_uniform_correlation_density_high_density_limit(self):
        np.testing.assert_allclose(
            xc.exponential_coulumb_uniform_correlation_density(
                density=jnp.array([1000., 10000.])
            ),
            [0., 0.],
            atol=1e-3
        )

    def test_exponential_coulomb_uniform_correlation_density_gradient(self):
        grad_fn = jax.vmap(
            jax.grad(xc.exponential_coulumb_uniform_correlation_density),
            in_axes=(0,)
        )
        density = jnp.linspace(0, 3, 11)
        alpha = 2
        beta = -1.00077
        gamma = 6.26099
        delta = -11.9041
        eta = 9.62614
        sigma = -1.48334
        nu = 1.

        y = jnp.pi * density / self.kappa
        denom = (gamma * y + delta * jnp.sqrt(y**3) + beta * jnp.sqrt(y)
            + sigma * jnp.sqrt(y ** 5) + eta * y ** 2
            + nu * jnp.pi * self.kappa**2 / self.amplitude * y ** 3
            + alpha)

        y_ddenom_dy = (gamma * y + 3 / 2. * delta * jnp.sqrt(y**3) + 1 / 2. * beta * jnp.sqrt(y)
            + 5 / 2. * sigma * jnp.sqrt(y ** 5) + 2 * eta * y ** 2
            + 3 * nu * jnp.pi * self.kappa**2 / self.amplitude * y ** 3
            )

        dEc_dy = -self.amplitude / (jnp.pi * denom ** 2) * (denom - y_ddenom_dy)
        dy_dn = jnp.pi / self.kappa

        dEc_dn = dEc_dy * dy_dn

        np.testing.assert_allclose(
            grad_fn(density), dEc_dn
        )

        # Gradient at density of 0.
        self.assertAlmostEqual(
            jax.grad(xc.exponential_coulumb_uniform_correlation_density)(0.),
            -self.amplitude / jnp.pi / alpha * dy_dn
        )

    @parameterized.parameters(1., 2., None, np.array([[1., 2., 3.]]))
    def test_get_lda_xc_energy_density_fn(self, params):
        # params is dummy argument so different values should not change the result
        lda_xc_energy_density_fn = xc.get_lda_xc_energy_density_fn()
        np.testing.assert_allclose(
            lda_xc_energy_density_fn(
                density=jnp.array([0., 1.]), params=params
            ),
            [0., -0.406069],
            atol=1e-5
        )

if __name__ == '__main__':
    absltest.main()