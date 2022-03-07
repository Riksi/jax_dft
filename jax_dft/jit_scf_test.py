"""Tests for jax_dft.jit_scf."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
from jax import tree_util
from jax.config import config
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np

from jax_dft import jit_scf
from jax_dft import neural_xc
from jax_dft import scf
from jax_dft import utils


# Set the default dtype as float64
config.update('jax_enable_x64', True)


class FlipAndAverageOnCentreTest(absltest.TestCase):
    def test_flip_and_average_on_centre(self):
        np.testing.assert_allclose(
            jit_scf._flip_and_average_on_centre(
                jnp.array([1., 2., 3.])
            ), [2., 2., 2.]
        )

    def test_flip_and_average_on_centre_fn(self):
        averaged_fn = jit_scf._flip_and_average_on_centre_fn(
                lambda x: jnp.array([4., 5., 6.])
            )
        np.testing.assert_allclose(
            averaged_fn(jnp.array([1., 2., 3.])),
            [5., 5., 5.]
        )


class ConnectionWeightsTest(parameterized.TestCase):

    @parameterized.parameters(
        (5, 2, [[1,   0.,  0.,  0.,  0. ],
                [0.5, 0.5, 0.,  0.,  0. ],
                [0.,  0.5, 0.5, 0.,  0. ],
                [0.,  0.,  0.5, 0.5, 0. ],
                [0.,  0.,  0.,  0.5, 0.5]
                ]),
        (5, 4, [[1,      0.,     0,      0,    0.  ],
                [0.5,    0.5,    0.,     0.,   0.  ],
                [0.33333334, 0.33333334, 0.33333334, 0.,   0.  ],
                [0.25,   0.25,   0.25,   0.25, 0.  ],
                [0.,     0.25,   0.25,   0.25, 0.25]
                ])
    )
    def test_connection_weights(self, num_iterations, num_mixing_iterations, expected_mask):
        np.testing.assert_allclose(
            jit_scf._connection_weights(num_iterations, num_mixing_iterations),
            expected_mask
        )


class KohnShamIterationTest(parameterized.TestCase):

    def setUp(self):
        super(KohnShamIterationTest, self).setUp()
        self.grids = jnp.linspace(-5, 5, 101)
        self.num_electrons = 2

    def _create_testing_initial_state(self, interaction_fn):
        locations = jnp.array([-0.5, 0.5])
        nuclear_charges = jnp.array([1, 1])
        return scf.KohnShamState(
            density=self.num_electrons * utils.gaussian(
                grids=self.grids, centre=0., sigma=1.
            ),
            total_energy=jnp.inf,
            locations=locations,
            nuclear_charges=nuclear_charges,
            grids=self.grids,
            external_potential=utils.get_atomic_chain_potential(
                grids=self.grids,
                locations=locations,
                nuclear_charges=nuclear_charges,
                interaction_fn=interaction_fn
            ),
            num_electrons=self.num_electrons
        )

    def _test_state(self, state, initial_state):
        # Next state density should normalise to num_electrons
        self.assertAlmostEqual(
            float(jnp.sum(state.density) * utils.get_dx(self.grids)),
            self.num_electrons
        )
        # Total energy should be finite after one iteration
        self.assertTrue(jnp.isfinite(state.total_energy))

        # These start out as `None`
        self.assertLen(state.hartree_potential, len(state.grids))
        self.assertLen(state.xc_potential, len(state.grids))

        # All these should remain unchanged
        np.testing.assert_allclose(initial_state.locations, state.locations)
        np.testing.assert_allclose(initial_state.nuclear_charges, state.nuclear_charges)
        np.testing.assert_allclose(initial_state.external_potential, state.external_potential)
        np.testing.assert_allclose(initial_state.grids, state.grids)
        self.assertEqual(initial_state.num_electrons, state.num_electrons)
        # gap = lumo - homo >= 0 since lumo >= homo
        # > 0 if you have an even number of electrons
        self.assertGreater(state.gap, 0)

    @parameterized.parameters(
        (utils.soft_coulomb, True),
        (utils.soft_coulomb, False),
        (utils.exponential_coulomb, True),
        (utils.exponential_coulomb, False)
    )
    def test_kohn_sham_iteration(self, interaction_fn, enforce_reflection_symmetry):
        initial_state = self._create_testing_initial_state(interaction_fn)
        next_state = scf.kohn_sham_iteration(
            state=initial_state,
            num_electrons=self.num_electrons,
            # 3d LDA exchange functional
            # with 0 correlation contribution
            xc_energy_density_fn=tree_util.Partial(
                lambda density: -0.73855 * density ** (1 / 3)
            ),
            interaction_fn=interaction_fn,
            enforce_reflection_symmetry=enforce_reflection_symmetry
        )
        self._test_state(next_state, initial_state)

    @parameterized.parameters(
        (utils.soft_coulomb, True),
        (utils.soft_coulomb, False),
        (utils.exponential_coulomb, True),
        (utils.exponential_coulomb, False)
    )
    def test_kohn_sham_iteration_neural_xc(self, interaction_fn, enforce_reflection_symmetry):
        init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
            stax.serial(
                stax.Dense(8), stax.Elu, stax.Dense(1)
            )
        )
        params_init = init_fn(random.PRNGKey(0))
        initial_state = self._create_testing_initial_state(interaction_fn)
        next_state = scf.kohn_sham_iteration(
            state=initial_state,
            num_electrons=self.num_electrons,
            # 3d LDA exchange functional
            # with 0 correlation contribution
            xc_energy_density_fn=tree_util.Partial(
                xc_energy_density_fn, params=params_init
            ),
            interaction_fn=interaction_fn,
            enforce_reflection_symmetry=enforce_reflection_symmetry
        )
        self._test_state(next_state, initial_state)

    def test_kohn_sham_iteration_neural_xc_energy_loss_gradient(self):
        # The network only has one layer.
        # The initial params contains weights with shape (1, 1) and bias (1,).
        init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
            stax.serial(stax.Dense(1)))
        init_params = init_fn(rng=random.PRNGKey(0))
        initial_state = self._create_testing_initial_state(
            utils.exponential_coulomb)
        target_energy = 2.
        spec, flatten_init_params = np_utils.flatten(init_params)

        def loss(flatten_params, initial_state, target_energy):
            state = scf.kohn_sham_iteration(
                state=initial_state,
                num_electrons=self.num_electrons,
                xc_energy_density_fn=tree_util.Partial(
                    xc_energy_density_fn,
                    params=np_utils.unflatten(spec, flatten_params)),
                interaction_fn=utils.exponential_coulomb,
                enforce_reflection_symmetry=True)
            return (state.total_energy - target_energy) ** 2

        grad_fn = jax.grad(loss)

        params_grad = grad_fn(
            flatten_init_params,
            initial_state=initial_state,
            target_energy=target_energy)

        np.testing.assert_allclose(params_grad, [-8.54995173, -14.75419501])
        # np.testing.assert_allclose(params_grad, [-8.549952, -14.754195])

        # Check whether the gradient values match the numerical gradient.
        np.testing.assert_allclose(
            optimize.approx_fprime(
                xk=flatten_init_params,
                f=functools.partial(
                    loss,
                    initial_state=initial_state,
                    target_energy=target_energy
                ), epsilon=1e-9
            ), params_grad, atol=2e-3
        )

    def test_kohn_sham_iteration_neural_xc_density_loss_gradient(self):
        # The network only has one layer.
        # The initial params contains weights with shape (1, 1) and bias (1,).
        init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
            stax.serial(stax.Dense(1)))
        init_params = init_fn(rng=random.PRNGKey(0))
        initial_state = self._create_testing_initial_state(
            utils.exponential_coulomb)
        target_density = (
                utils.gaussian(grids=self.grids, centre=-0.5, sigma=1.)
                + utils.gaussian(grids=self.grids, centre=0.5, sigma=1.))
        spec, flatten_init_params = np_utils.flatten(init_params)

        def loss(flatten_params, initial_state, target_density):
            state = scf.kohn_sham_iteration(
                state=initial_state,
                num_electrons=self.num_electrons,
                xc_energy_density_fn=tree_util.Partial(
                    xc_energy_density_fn,
                    params=np_utils.unflatten(spec, flatten_params)),
                interaction_fn=utils.exponential_coulomb,
                enforce_reflection_symmetry=False)
            return jnp.sum(jnp.abs(state.density - target_density)) * utils.get_dx(
                self.grids)

        grad_fn = jax.grad(loss)

        params_grad = grad_fn(
            flatten_init_params,
            initial_state=initial_state,
            target_density=target_density)

        # Check gradient values.
        np.testing.assert_allclose(params_grad, [-1.3413697, 0.], atol=5e-7)
        # np.testing.assert_allclose(params_grad, [-1.34137017, 0.], atol=5e-7)

        # Check whether the gradient values match the numerical gradient.
        np.testing.assert_allclose(
            optimize.approx_fprime(
                xk=flatten_init_params,
                f=functools.partial(
                    loss,
                    initial_state=initial_state,
                    target_density=target_density
                ), epsilon=1e-9
            ), params_grad, atol=2e-4
        )

    def test_kohn_sham_iteration_neural_xc_density_loss_gradient_symmetry(self):
        # The network only has one layer.
        # The initial params contains weights with shape (1, 1) and bias (1,).
        init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
            stax.serial(stax.Dense(1)))
        init_params = init_fn(rng=random.PRNGKey(0))
        initial_state = self._create_testing_initial_state(
            utils.exponential_coulomb)
        target_density = (
                utils.gaussian(grids=self.grids, centre=-0.5, sigma=1.)
                + utils.gaussian(grids=self.grids, centre=0.5, sigma=1.))
        spec, flatten_init_params = np_utils.flatten(init_params)

        def loss(flatten_params, initial_state, target_density):
            state = scf.kohn_sham_iteration(
                state=initial_state,
                num_electrons=self.num_electrons,
                xc_energy_density_fn=tree_util.Partial(
                    xc_energy_density_fn,
                    params=np_utils.unflatten(spec, flatten_params)),
                interaction_fn=utils.exponential_coulomb,
                enforce_reflection_symmetry=True)
            return jnp.sum(jnp.abs(state.density - target_density)) * utils.get_dx(
                self.grids)

        grad_fn = jax.grad(loss)

        params_grad = grad_fn(
            flatten_init_params,
            initial_state=initial_state,
            target_density=target_density)

        # Check gradient values.
        np.testing.assert_allclose(params_grad, [-1.3413697, 0.], atol=5e-7)
        # np.testing.assert_allclose(params_grad, [-1.34137017, 0.], atol=5e-7)

        # Check whether the gradient values match the numerical gradient.
        np.testing.assert_allclose(
            optimize.approx_fprime(
                xk=flatten_init_params,
                f=functools.partial(
                    loss,
                    initial_state=initial_state,
                    target_density=target_density
                ), epsilon=1e-9
            ), params_grad, atol=1e-3
        )
