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

    def test_self_interaction_layer_one_electron(self):
        # [11]
        grids = jnp.linspace(-5, 5, 11)
        # [11]
        density = utils.gaussian(grids=grids, centre=1., sigma=1.)
        # [1, 11, 1]
        reshaped_density = density[jnp.newaxis, :, jnp.newaxis]

        init_fn, apply_fn = neural_xc.self_interaction_layer(
            grids, utils.exponential_coulomb
        )

        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=((-1, 11, 1), (-1, 11, 1))
        )
        self.assertEqual(output_shape, (-1, 11, 1))
        self.assertAlmostEqual(init_params, (1.,))
        np.testing.assert_allclose(
            # beta = exp(- ((sum(density) * dx - 1) / width)^2 ) ~ 1
            # Since density is Gaussian, sum(density) * dx ~ 1
            # so the beta ~ 1
            # (see previous test for more details)
            # Hence output is
            # -0.5 * hartree_potential * beta + features * (1 - beta)
            # ~ -0.5 * hartree_potential
            apply_fn(init_params, (reshaped_density, jnp.ones_like(reshaped_density))),
            -0.5 * scf.get_hartree_potential(
                density=density,
                grids=grids,
                interaction_fn=utils.exponential_coulomb)[jnp.newaxis, :, jnp.newaxis]
            )

    def test_self_interaction_layer_large_num_electrons(self):
        # [11]
        grids = jnp.linspace(-5, 5, 11)
        # [11]
        density = 100. * utils.gaussian(grids=grids, centre=1., sigma=1.)
        # [1, 11, 1]
        reshaped_density = density[jnp.newaxis, :, jnp.newaxis]
        features = np.random.rand(*reshaped_density.shape)

        init_fn, apply_fn = neural_xc.self_interaction_layer(
            grids, utils.exponential_coulomb
        )

        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=((-1, 11, 1), (-1, 11, 1))
        )
        self.assertEqual(output_shape, (-1, 11, 1))
        self.assertAlmostEqual(init_params, (1.,))
        np.testing.assert_allclose(
            # beta = dexp(-((sum(density)*dx - 1) / width)^2 ) ~ 1
            # Since density is 100 times a Gaussian, sum(density)*dx ~ 100,
            # since width=1, the exponential term is exp(-99^2) ~ 0
            # => beta ~ 0
            # (see previous test for more details)
            # Hence output is
            # -0.5 * hartree_potential * beta + features * (1 - beta)
            # ~ features
            apply_fn(init_params, (reshaped_density, features)),
            features
            )

    def test_linear_interpolation(self):
        init_fn, apply_fn = neural_xc.linear_interpolation()
        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 3, 2)
        )
        self.assertEqual(output_shape, (-1, 5, 2))
        self.assertEmpty(init_params)

        # samples at linspace(0, inputs.size - 1, num=new_size)
        # for each element along dimension one hence inputs.size=3
        # -> linspace(0, 2, num=5) = [0., 0.5, 1., 1.5, 2.]
        np.testing.assert_allclose(
            apply_fn(
                (), jnp.array([[[1, 2], [3, 4], [5, 6]]], dtype=float),
            ),
            [[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]]
        )

    def test_linear_interpolation_transpose(self):
        init_fn, apply_fn = neural_xc.linear_interpolation_transpose()
        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 5, 2)
        )
        self.assertEqual(output_shape, (-1, 3, 2))
        self.assertEmpty(init_params)

        np.testing.assert_allclose(
            apply_fn((), jnp.array([[[1.0, 2], [3, 4], [5, 6], [7, 8], [9, 10]]])),
            [[[1.5, 2.5], [5, 6], [8.5, 9.5]]]
        )

    def test_upsampling_block(self):
        init_fn, apply_fn = neural_xc.upsampling_block(32, 'softplus')
        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 9, 1)
        )
        self.assertEqual(output_shape, (-1, 17, 32))

        output = apply_fn(
            init_params, jnp.array(np.random.randn(6, 9, 1))
        )

        self.assertEqual(output.shape, (6, 17, 32))

    @parameterized.parameters('relu', 'elu', 'softplus', 'swish')
    def test_downsampling_block(self, activation):
        init_fn, apply_fn = neural_xc.downsampling_block(32, activation)
        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 9, 1)
        )
        self.assertEqual(output_shape, (-1, 5, 32))

        output = apply_fn(
            init_params, jnp.array(np.random.randn(6, 9, 1))
        )

        self.assertEqual(output.shape, (6, 5, 32))

    def test_build_unet(self):
        init_fn, apply_fn = neural_xc.build_unet(
            num_filters_list=[2, 4, 8],
            core_num_filters=16,
            activation='softplus'
        )
        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 9, 1)
        )
        self.assertEqual(output_shape, (-1, 9, 1))

        output = apply_fn(
            init_params, jnp.array(np.random.randn(6, 9, 1))
        )

        self.assertEqual(output.shape, (6, 9, 1))

    def test_wrap_self_interaction_layer_one_electron(self):
        # [9]
        grids = jnp.linspace(-5, 5, 9)
        # [9]
        density = utils.gaussian(grids=grids, centre=1., sigma=1.)
        # [1, 9, 1]
        reshaped_density = density[jnp.newaxis, :, jnp.newaxis]
        init_fn, apply_fn = neural_xc.wrap_network_with_self_interaction_layer(
            network=neural_xc.build_unet(
                num_filters_list=[2, 4],
                core_num_filters=4,
                activation='swish'
            ),
            grids=grids,
            interaction_function=utils.exponential_coulomb
        )
        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 9, 1)
        )
        self.assertEqual(output_shape, (-1, 9, 1))
        np.testing.assert_allclose(
            # Output will be the almost identical to input since only one electron
            # (see `test_self_interaction_layer_one_electron` for details)
            apply_fn(init_params, reshaped_density),
            -0.5 * scf.get_hartree_potential(
                density=density,
                grids=grids,
                interaction_fn=utils.exponential_coulomb)[jnp.newaxis, :, jnp.newaxis]
        )

    def test_wrap_self_interaction_layer_large_num_electrons(self):
        # [9]
        grids = jnp.linspace(-5, 5, 9)
        # [9]
        density = 100 * utils.gaussian(grids=grids, centre=1., sigma=1.)
        # [1, 9, 1]
        reshaped_density = density[jnp.newaxis, :, jnp.newaxis]
        inner_network_init_fn,  inner_network_apply_fn = neural_xc.build_unet(
            num_filters_list=[2, 4],
            core_num_filters=4,
            activation='swish'
        )
        inner_network_output_shape, inner_network_init_params = inner_network_init_fn(
            random.PRNGKey(0), input_shape=(-1, 9, 1)
        )
        init_fn, apply_fn = neural_xc.wrap_network_with_self_interaction_layer(
            network=(inner_network_init_fn, inner_network_apply_fn),
            grids=grids,
            interaction_function=utils.exponential_coulomb
        )
        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 9, 1)
        )
        self.assertEqual(output_shape, (-1, 9, 1))
        np.testing.assert_allclose(
            # Output will be the almost identical to unet features since many electrons
            # (see `test_self_interaction_layer_large_num_electrons` for details)
            apply_fn(init_params, reshaped_density),
            inner_network_apply_fn(init_params[1][1], reshaped_density)
        )

    def test_build_sliding_net(self):
        init_fn, apply_fn = neural_xc.build_sliding_net(
            window_size=3,
            num_filters_list=[2, 4, 8],
            activation='softplus'
        )
        output_shape, init_params = init_fn(
            random.PRNGKey(0), input_shape=(-1, 9, 1)
        )
        self.assertEqual(output_shape, (-1, 9, 1))

        output = apply_fn(
            init_params, jnp.array(np.random.randn(6, 9, 1))
        )

        self.assertEqual(output.shape, (6, 9, 1))

    def test_build_sliding_net_invalid_window_size(self):
        with self.assertRaisesRegex(
                ValueError,
                'window size cannot be less than 1 but got 0'
        ):
            neural_xc.build_sliding_net(
                window_size=0,
                num_filters_list=[2, 4, 8],
                activation='softplus'
            )














class GlobalFunctionalTest(parameterized.TestCase):

    def test_spatial_shift_input(self):
        np.testing.assert_allclose(
            neural_xc._spatial_shift_input(
                # [1, 5, 2]
                features=jnp.array([[
                    [11., 21.], [12., 22.], [13., 23.], [14., 24.], [15., 25.]
                ]]),
                num_spatial_shift=4
            ),
            # [4, 5, 2]
            [
                # offset = 0
                [[11., 21.], [12., 22.], [13., 23.], [14., 24.], [15., 25.]],
                # offset = 1
                [[12., 22.], [13., 23.], [14., 24.], [15., 25.], [0., 0.]],
                # offset = 2
                [[13., 23.], [14., 24.], [15., 25.], [0., 0.], [0., 0.]],
                # offset = 3
                [[14., 24.], [15., 25.], [0., 0.], [0., 0.], [0., 0.]],
            ]
        )

    def test_reverse_spatial_shift_output(self):
        np.testing.assert_allclose(
            neural_xc._reverse_spatial_shift_output(
                # [1, 5]
                array=jnp.array([
                    [1., 2., 3., 4., 5.],
                    [12., 13., 14., 15., 0.],
                    [23., 24., 25., 0., 0.],
                ]
             )
            ),
            # [1, 5]
            [
                # offset = 0
                [1., 2., 3., 4., 5.],
                # offset = -1
                [0, 12., 13., 14, 15.],
                # offset = -2
                [0., 0., 23., 24., 25.],
            ]
        )

    @parameterized.parameters(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    def test_is_power_of_two_true(self, number):
        self.assertTrue(neural_xc._is_power_of_two(number))

    @parameterized.parameters(0, 3, 6, 9)
    def test_is_power_of_two_false(self, number):
        self.assertFalse(neural_xc._is_power_of_two(number))


