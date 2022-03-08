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

    @parameterized.parameters(True, False)
    def test_kohn_sham_iteration_neural_xc(self, enforce_reflection_symmetry):
        init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
            stax.serial(
                stax.Dense(8), stax.Elu, stax.Dense(1)
            )
        )
        params_init = init_fn(random.PRNGKey(0))
        initial_state = self._create_testing_initial_state(utils.exponential_coulomb)
        next_state = jit_scf.kohn_sham_iteration(
            state=initial_state,
            num_electrons=self.num_electrons,
            # 3d LDA exchange functional
            # with 0 correlation contribution
            xc_energy_density_fn=tree_util.Partial(
                xc_energy_density_fn, params=params_init
            ),
            interaction_fn=utils.exponential_coulomb,
            enforce_reflection_symmetry=enforce_reflection_symmetry
        )
        self._test_state(next_state, initial_state)


class KohnShamTest(parameterized.TestCase):

    def setUp(self):
        super(KohnShamTest, self).setUp()
        self.grids = jnp.linspace(-5, 5, 101)
        self.num_electrons = 2
        self.locations = jnp.array([-0.5, 0.5])
        self.nuclear_charges = jnp.array([1, 1])

    def _create_testing_external_potential(self, interaction_fn):
        return utils.get_atomic_chain_potential(
            grids=self.grids,
            locations=self.locations,
            nuclear_charges=self.nuclear_charges,
            interaction_fn=interaction_fn
        )

    def _test_state(self, state, external_potential):
        # The density in  the final state should normalise
        # to the number of electrons
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
        np.testing.assert_allclose(state.locations, self.locations)
        np.testing.assert_allclose(state.nuclear_charges, self.nuclear_charges)
        np.testing.assert_allclose(external_potential, state.external_potential)
        np.testing.assert_allclose(state.grids, self.grids)
        self.assertEqual(state.num_electrons, self.num_electrons)
        # gap = lumo - homo >= 0 since lumo >= homo
        # > 0 if you have an even number of electrons
        self.assertGreater(state.gap, 0)

    @parameterized.parameters(
        (jnp.inf, [False, True, True]),
        (-1., [False, False, False])
    )
    def test_kohn_sham_neural_xc_density_mse_converge_tolerance(self, density_mse_converge_tolerance, expected_converged):
        # See notes in jit_scf._kohn_sham_iteration as to why here first element is False
        # in the first case whilst all were true for scf.kohn_sham
        # (Basically the first element of the result
        # is the initial state with converged=False)
        init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
            stax.serial(
                stax.Dense(8), stax.Elu, stax.Dense(1)
            )
        )
        params_init = init_fn(random.PRNGKey(0))
        state = jit_scf.kohn_sham(
            locations=self.locations,
            nuclear_charges=self.nuclear_charges,
            num_electrons=self.num_electrons,
            num_iterations=3,
            grids=self.grids,
            # 3d LDA exchange functional, zero correlation functional
            xc_energy_density_fn=tree_util.Partial(xc_energy_density_fn,
                                                   params=params_init),
            interaction_fn=utils.exponential_coulomb,
            # Note that we have to provide an initial value for density for jit
            # whereas in the other version there was the option of initialising
            # it within kohn_sham
            initial_density=self.num_electrons * utils.gaussian(
                self.grids, centre=0., sigma=0.5
            ),
            density_mse_converge_tolerance=density_mse_converge_tolerance
        )
        np.testing.assert_array_equal(
            state.converged, expected_converged
        )

        for single_state in scf.state_iterator(state):
            self._test_state(
                single_state,
                self._create_testing_external_potential(utils.exponential_coulomb)
            )


    @parameterized.parameters(2, 3, 4, 5)
    def test_kohn_sham_neural_xc_num_mixing_iterations(self, num_mixing_iterations):
        # See notes in jit_scf._kohn_sham_iteration as to why here first element is False
        # in the first case whilst all were true for scf.kohn_sham
        # (Basically the first element of the resul
        # is the initial state with converged=False)
        init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
            stax.serial(
                stax.Dense(8), stax.Elu, stax.Dense(1)
            )
        )
        params_init = init_fn(random.PRNGKey(0))
        state = jit_scf.kohn_sham(
            locations=self.locations,
            nuclear_charges=self.nuclear_charges,
            num_electrons=self.num_electrons,
            num_iterations=3,
            grids=self.grids,
            # 3d LDA exchange functional, zero correlation functional
            xc_energy_density_fn=tree_util.Partial(xc_energy_density_fn,
                                                   params=params_init),
            interaction_fn=utils.exponential_coulomb,
            # Note that we have to provide an initial value for density for jit
            # whereas in the other version there was the option of initialising
            # it within kohn_sham
            initial_density=self.num_electrons * utils.gaussian(
                self.grids, centre=0., sigma=0.5
            ),
            num_mixing_iterations=num_mixing_iterations
        )
        for single_state in scf.state_iterator(state):
            self._test_state(
                single_state,
                self._create_testing_external_potential(utils.exponential_coulomb)
            )


if __name__ == '__main__':
    absltest.main()