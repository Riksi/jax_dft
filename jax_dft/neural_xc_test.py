from absl.testing import absltest
from absl.testing import parameterized
from jax import random
from jax.config import config
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np

from jax_dft import neural_xc
from jax_dft import scf
from jax_dft import utils

# Set the default dtype as float64
config.update('jax_enable_x64', False)


class NetworkTest(parameterized.TestCase):

    def test_negativity_transform(self):
        init_fn, apply_fn = neural_xc.negativity_transform()
        output_shape, init_params = init_fn(
            random.PRNGKey(0),
            input_shape=(-1, 3, 1)
        )

        self.assertEqual(output_shape, (-1, 3, 1))
        self.assertEqual(init_params, ())
        self.assertTrue(
            np.all(
                apply_fn(
                    init_params,
                    # Not sure where the 0.278 comes from
                    # since swish([-0.5, 0., 0.5]
                    # = [-0.31122967, 0., 0.18877033]
                    jnp.array(
                        [[[-0.5], [0.], [0.5]]]
                    )
                ) <= 0.278
            )
        )

    @parameterized.parameters(0.5, 1.5, 2.)
    def test_exponential_function(self, width):
        # Assuming _exponential_function is correct,
        # this is numerical integral of 1/(2*w) * exp(-x/w)
        # between -20.48 to 20.48
        # at intervals of size 0.08 i.e. dx = 0.08.
        # It will be roughly equal to 1
        # since exp(-x/w) goes to zero at the endpoints.
        grids = np.arange(-256, 257) * 0.08
        self.assertAlmostEqual(
            jnp.sum(neural_xc._exponential_function(grids, width)) * 0.08,
            1.,
            places=2
        )

    def test_exponential_function_channels(self):
        self.assertEqual(
            neural_xc._exponential_function_channels(
                displacements=jnp.array(np.random.rand(11, 11)),
                widths=jnp.array([1., 2., 3.])
            ).shape, (11, 11, 3)
        )

    def test_exponential_global_convolution(self):
        init_fn, apply_fn = neural_xc.exponential_global_convolution(
            num_channels=2, grids=jnp.linspace(-1, 1, 5),
            minval=0.1, maxval=2.
        )

        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 5, 1)
        )
        output = apply_fn(
            init_params,
            jnp.array(np.random.rand(1, 5, 1))
        )

        self.assertEqual(output_shape, (-1, 5, 2))
        self.assertLen(init_params, 1)
        self.assertEqual(init_params[0].shape, (2,))
        self.assertEqual(output.shape, (1, 5, 2))


class TempTest(parameterized.TestCase):
    @parameterized.parameters((0.5,  0.77880025), (1., 1.), (100., 0.))
    def test_self_interaction_weight(self, density_integral, expected_weight):
        grids = jnp.linspace(-5, 5, 11)
        # self_interaction_weight first numerically integrates the density to get `I`
        # then it returns exp(-((I - 1)/width)^2)
        # Here we have
        # I = density_integral * sum_x(gaussian(x, mu, sigma)) ~ density_integral
        # since even though the interval is not symmetrical about mu it is fairly
        # large so the numerical integral is approximately 1.
        # width=1. so result ~ exp(-(density_integral - 1)^2)
        # density_integral = 0.5 -> exp(-(0.5 - 1)^2) = exp(-0.25) ~ 0.77880078
        #   but here we actually need to use the close value of 0.77880025
        # density_integral = 1. -> exp(-(1 - 1)^2) = 1.
        # density_integral = 100. -> exp(-(100 - 1)^2) ~ 0
        self.assertAlmostEqual(
            neural_xc.self_interaction_weight(
                # [1, 11, 1]
                density_integral * utils.gaussian(
                    grids=grids, centre=1., sigma=1.
                )[jnp.newaxis, :, jnp.newaxis],
                dx=utils.get_dx(grids),
                width=1.
            ), expected_weight
        )







class GlobalFunctionalTest(parameterized.TestCase):

    @parameterized.parameters(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    def test_is_power_of_two_true(self, number):
        self.assertTrue(neural_xc._is_power_of_two(number))

    @parameterized.parameters(0, 3, 6, 9)
    def test_is_power_of_two_false(self, number):
        self.assertFalse(neural_xc._is_power_of_two(number))