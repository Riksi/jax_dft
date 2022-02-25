import functools
import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import tree_util
from jax import random
from jax.config import config
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

from jax_dft import neural_xc
from jax_dft import utils
from jax_dft import np_utils
from jax_dft import scf

config.update('jax_enable_x64', True)

class ScfTest(parameterized.TestCase):

    def test_discrete_laplacian(self):
        np.testing.assert_allclose(
            scf.discrete_laplacian(6),
            [
                [-5. / 2, 4. / 3, -1. / 12, 0., 0., 0.],
                [4. / 3, -5. / 2, 4. / 3, -1. / 12, 0., 0.],
                [-1. / 12, 4. / 3, -5. / 2, 4. / 3, -1. / 12, 0.],
                [0., -1. / 12, 4. / 3, -5. / 2, 4. / 3, -1. / 12],
                [0., 0., -1. / 12, 4. / 3, -5. / 2, 4. / 3],
                [0., 0., 0., -1. / 12, 4. / 3, -5. / 2],
            ],
            atol=1e-6
        )

    def test_get_kinetic_matrix(self):
        np.testing.assert_allclose(
            scf.get_kinetic_matrix(grids=jnp.linspace(-10, 10, 6)),
            [
                [0.078125, -0.04166667, 0.00260417, 0., 0., 0.],
                [-0.04166667, 0.078125, -0.04166667, 0.00260417, 0., 0.],
                [0.00260417, -0.04166667, 0.078125, -0.04166667, 0.00260417, 0.],
                [0., 0.00260417, -0.04166667, 0.078125, -0.04166667, 0.00260417],
                [0., 0., 0.00260417, -0.04166667, 0.078125, -0.04166667],
                [0., 0., 0., 0.00260417, -0.04166667, 0.078125],
            ],
            atol=1e-6
        )

    @parameterized.parameters(
        (1, -1.),  # total_eigen_energies = -1.
        (2, -2.),  # total_eigen_energies = -1. - 1.
        (3, 0.),  # total_eigen_energies = -1. - 1. + 2.
        (4, 2.),  # total_eigen_energies = -1. - 1. + 2. + 2.
        (5, 7.),  # total_eigen_energies = -1. - 1. + 2. + 2. + 5.
        (6, 12.),  # total_eigen_energies = -1. - 1. + 2. + 2. + 5. + 5.
    )
    def test_get_total_eigen_energies(self, num_electrons, expected_total_eigen_energies):
        self.assertAlmostEqual(
            scf.get_total_eigen_energies(num_electrons, jnp.array([-1., 2., 5.])),
            expected_total_eigen_energies
        )

    @parameterized.parameters(
        # 1. Only first wavefunction used [0., 0., 1., 0., 0.]
        # (actually function does the step below for both wavefunctions
        #  but discards the second one)
        # -> [0., 0., 1., 0., 0.] / sqrt(1. * 0.1)
        # denominator is sqrt of approximate integral of abs(Psi)^2
        # with dx=0.1, the distance between consecutive points
        # on the grid
        # -> [0, 0., 10., 0., 0.]
        # square of the normalised wavefunction, no repeats
        (1, [0, 0., 10., 0., 0.]),
        # 2. Normalised Psi repeated twice, then summed
        (2, [0, 0., 20., 0., 0.]),
        # 3. Second wavefunction also used
        # [0., -1., 0., 1., 0.]
        # -> [0., 1., 0., 1., 0.] / sqrt(2. * 0.1)
        # -> [0., 5., 0., 5., 0.]
        (3, [0, 5., 20., 5., 0.]),
        # 4. Second wavefunction also repeated
        (4, [0, 10., 20., 10., 0.]),
    )
    def test_wavefunctions_to_density(self, num_electrons, expected_density):
        np.testing.assert_allclose(
            scf.wavefunctions_to_density(
                num_electrons=num_electrons,
                wavefunctions=jnp.array([
                    [0., 0., 1., 0., 0.],
                    [0., -1., 0., 1., 0.],
                ]),
                grids=jnp.arange(5) * 0.1
            ),
            expected_density
        )


    @parameterized.parameters(
        (1, 0.),  # gap = -1. - (-1.)
        (2, 3.),  # gap = 2. - (-1.)
        (3, 0.),  # gap = 2. - 2.
        (4, 7.),  # gap = 9. - 2.
        (5, 0.),  # gap = 9. - 9.
        (6, 78.),  # gap = 87. - 9.
    )
    def test_get_gap(self, num_electrons, expected_gap):
        self.assertAlmostEqual(
            scf.get_gap(num_electrons, jnp.array([-1., 2., 9., 87.])),
            expected_gap
        )

    @parameterized.parameters(
        utils.soft_coulomb,
        utils.exponential_coulomb
    )
    def test_get_hartree_energy(self, interaction_fn):
        # Harmonic oscillator (m = omega = hbar = 1)
        # En = n + 1/2, n = 0, 1, ...
        grids = jnp.linspace(-5, 5, 11)
        dx = utils.get_dx(grids)
        density = utils.gaussian(grids, centre=1., sigma=1.)

        # This is done in vectorised way in the function
        # so for test use nested for loops
        expected_hartree_energy = 0.
        for x0, n0 in zip(grids, density):
            for x1, n1 in zip(grids, density):
                expected_hartree_energy += 0.5 * n0 * n1 * interaction_fn(x0 - x1) * dx ** 2

        self.assertAlmostEqual(
            float(scf.get_hartree_energy(density, grids, interaction_fn)),
            float(expected_hartree_energy)
        )

    @parameterized.parameters(
        utils.soft_coulomb,
        utils.exponential_coulomb
    )
    def test_get_hartree_potential(self, interaction_fn):
        # Harmonic oscillator (m = omega = hbar = 1)
        # En = n + 1/2, n = 0, 1, ...
        grids = jnp.linspace(-5, 5, 11)
        dx = utils.get_dx(grids)
        density = utils.gaussian(grids, centre=1., sigma=1.)

        # This is done in vectorised way in the function
        # so for test use nested for loops
        expected_hartree_potential = np.zeros_like(grids)
        for i, x0 in enumerate(grids):
            for x1, n1 in zip(grids, density):
                expected_hartree_potential[i] += np.sum(0.5 * n1 * interaction_fn(x0 - x1)) * dx

        np.testing.assert_allclose(
            scf.get_hartree_potential(density, grids, interaction_fn),
            expected_hartree_potential
        )

    def test_get_external_potential_energy(self):
        grids = jnp.linspace(-5, 5, 10001)

        # exp(-x^2) * exp(-(x - 1)^2) = exp(-2x^2 + 2x - 1)
        # = exp(-2*(x^2 - x + 1/4) - 1/2)
        # = exp(-1/2)*exp(-2*(x - 1/2)^2)
        # = exp(-1/2)*exp(-(x - 1/2)^2/(2 * (1/2)^2))
        # mu = 1/2, sigma = 1/2
        # => integral between -oo to oo = exp(-1/2) * (sqrt(2 * pi) * 1/2)
        # = = exp(-1/2) * (pi / sqrt(2)) =  sqrt(pi / (2 * e))
        self.assertAlmostEqual(
            float(
                scf.get_external_potential_energy(
                    external_potential=-jnp.exp(-grids ** 2),
                    density=jnp.exp(-(grids - 1) ** 2),
                    grids=grids
                )
            ),
            # Analytical integral
            -np.sqrt(jnp.pi / (2 * np.e))
        )

    def test_get_xc_energy(self):
        grids = jnp.linspace(-5, 5, 10001)
        # exchange energy = -0.73855 \int n^(4 / 3) dx
        # exchange potential = -0.73855 * (4 / 3) * n^(1 / 3)
        # 3D LDA exchange functional
        # Energy is \int -0.73855 * n ** (1 / 3) * n dx
        # = \int -0.73855 * n ** (4 / 3) dx
        xc_energy_density_function = lambda density: -0.73855 * density ** (1 / 3)
        density = jnp.exp(-(grids - 1) ** 2)
        self.assertAlmostEqual(
            scf.get_xc_energy(density, xc_energy_density_function, grids),
            # n^(4 / 3) = exp(-(4 / 3)*(x - 1))
            # = exp(-(x - 1) / (2 * 3 / 8))
            # mu = 1, sigma = sqrt(3 / 8)
            # -0.73855 * int n^(4 / 3) dx
            # = -0.73855 * sqrt(2*pi) * sigma
            # = -0.73855 * sqrt(3*pi) / 2
            # = -1.13367
            -1.13367,
            places=5
        )

    def test_get_xc_potential(self):
        grids = jnp.linspace(-5, 5, 10001)
        # exchange energy = -0.73855 \int n^(4 / 3) dx
        # exchange potential = -0.73855 * (4 / 3) * n^(1 / 3)
        # 3D LDA exchange functional
        # Energy is \int -0.73855 * n ** (1 / 3) * n dx
        # = \int -0.73855 * n ** (4 / 3) dx
        xc_energy_density_function = lambda density: -0.73855 * density ** (1 / 3)
        density = jnp.exp(-(grids - 1) ** 2)
        np.testing.assert_allclose(
            scf.get_xc_potential(density, xc_energy_density_function, grids),
            -0.73855 * (4 / 3) * density ** (1 / 3)
        )

    def test_get_xc_potential_hartree(self):
        grids = jnp.linspace(-5, 5, 10001)
        density = utils.gaussian(grids=grids, centre=1., sigma=1.)

        # get_hartree_potential returns
        # v(x) = 0.5 \int n(x') * f(x' - x) dx'
        # where f is an interaction function

        # hhp(x) = 0.25 *  \int n(x') * f(x' - x) dx'

        # XC energy is \int f(x) * n(x) dx
        # = 0.5 * \int \int n(x') * n(x) *  f(x' - x) dx' dx

        # XC potential is
        # 0.25 * \int n(x') *  f(x' - x) dx'
        # + 0.25 * \int n(x) *  f(x' - x)dx

        # The interaction_fn is exponential_coulomb
        #  A * exp(-κ * |x - x'|)
        # which is symmetrical so we
        # we can relabel x -> x', x' -> x in the second term
        # which makes both terms identical

        # Hence XC potential is
        # 0.5 * \int n(x') *  f(x - x') dx'
        # = 0.5 * \int n(x') *  A * exp(-κ * |x - x'|) dx'

        # But this is exactly the result returned by
        # get_hartree_potential

        def half_hartree_potential(density):
            return 0.5 * scf.get_hartree_potential(
                density,
                grids,
                interaction_fn=utils.exponential_coulomb
            )

        np.testing.assert_allclose(
            scf.get_xc_potential(density, half_hartree_potential, grids),
            scf.get_hartree_potential(density, grids, utils.exponential_coulomb)
        )


class KohnShamIterationTest(parameterized.TestCase):

    def setUp(self):
        super(KohnShamIterationTest, self).setUp()
        self.grids = jnp.linspace(-5, 5, 101)
        self.num_electrons = 2

    def _create_initial_testing_state(self, interaction_fn):
        locations = jnp.array([0.5, 0.5])
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
        initial_state = self._create_initial_testing_state(interaction_fn)
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
        initial_state = self._create_initial_testing_state(interaction_fn)
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


class GetInitialDensityTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.states = scf.KohnShamState(
            density=np.random.random((5, 100)),
            total_energy=np.random.random((5,)),
            locations=np.random.random((5, 2)),
            nuclear_charges=np.random.random((5, 2)),
            external_potential=np.random.random((5, 100)),
            grids=np.random.random((5, 100)),
            num_electrons=np.random.randint(10, size=5)
        )

    def test_get_initial_density_exact(self):
        np.testing.assert_allclose(
            scf.get_initial_density(self.states, 'exact'),
            self.states.density
        )

    def test_get_initial_density_unknown(self):
        with self.assertRaisesRegex(
                ValueError, "Unknown initialisation method init"
        ):
            scf.get_initial_density(self.states, 'init')













