from absl.testing import absltest
from absl.testing import parameterized
from jax.config import config
import jax.numpy as jnp
import numpy as np

from jax_dft import constants
from jax_dft import utils


# Set the default dtype as float64
config.update('jax_enable_x64', True)


class UtilsTest(parameterized.TestCase):
    @parameterized.parameters(
        (2, [4., 1., 0., 0.]),
        (-2, [0., 0., 1., 2.])
    )
    def test_shift(self, offset, expected_output):
        np.testing.assert_allclose(
            utils.shift(jnp.array([1., 2., 4., 1.]), offset=offset),
            expected_output
        )

    def test_dx(self):
        self.assertAlmostEqual(
            utils.get_dx( jnp.linspace(0, 1, 11)), 0.1
        )

    def test_dx_incorrect_dim(self):
        with self.assertRaisesRegex(
            ValueError, "grids.ndim is expected to be 1 but got 2"
        ):
            utils.get_dx(jnp.array([[-0.1], [0.], [0.1]]))

    @parameterized.parameters(
        (0., 2., 1 / (np.sqrt(2 * np.pi) * 2)),
        (3., 0.5, 1 / (np.sqrt(2 * np.pi) * 0.5)),
    )
    def test_gaussian(self, centre, sigma, expected_max_value):
        gaussian = utils.gaussian(
            grids=jnp.linspace(-10, 10, 201),
            centre=centre,
            sigma=sigma
        )
        self.assertAlmostEqual(float(jnp.sum(gaussian) * 0.1), 1, places=5)
        self.assertAlmostEqual(float(jnp.amax(gaussian)), expected_max_value)

    @parameterized.parameters(-1., 0., 1.)
    def test_soft_coulomb(self, center):
        grids = jnp.linspace(-10, 10, 201)
        soft_coulomb_interaction = utils.soft_coulomb(grids - center)

        # It is 1 / sqrt((x - center) ** 2 + 1)
        # When x = center, denominator is 1 so value is 1
        self.assertAlmostEqual(
            float(jnp.amax(soft_coulomb_interaction)),
            1
        )

        # The position where function is maximised is for x = center
        self.assertAlmostEqual(
            float(grids[jnp.argmax(soft_coulomb_interaction)]),
            center
        )

    @parameterized.parameters(-1., 0., 1.)
    def test_exponential_coulomb(self, center):
        grids = jnp.linspace(-10, 10, 201)
        soft_coulomb_interaction = utils.exponential_coulomb(grids - center)

        # It is A * exp(-κ * |x - center|)
        # Max when x = center where value is A
        self.assertAlmostEqual(
            float(jnp.amax(soft_coulomb_interaction)),
            constants.EXPONENTIAL_COULOMB_AMPLITUDE
        )

        # The position where function is maximised is for x = center
        self.assertAlmostEqual(
            float(grids[jnp.argmax(soft_coulomb_interaction)]),
            center
        )

    def test_get_atomic_chain_potential_soft_coulomb(self):
        potential = utils.get_atomic_chain_potential(
            grids=jnp.linspace(-10, 10, 201),
            locations=jnp.array([0., 1.]),
            nuclear_charges=jnp.array([2, 1]),
            interaction_fn=utils.soft_coulomb
        )
        # potential[i] = -sum_a(Z[a] / sqrt((x[0] - X[a])**2 + 1))
        # x[0] = -10 -> p[0] = -2 / sqrt((-10 - 0)**2 + 1) - 1 / sqrt((-10 - 1)**2 + 1)
        # = -2 / sqrt(101) - 1/ sqrt(122)
        self.assertAlmostEqual(float(potential[0]), -0.2895432)
        # x[100] = 0 -> p[100] = -2 / sqrt((0 - 0)**2 + 1) - 1 / sqrt((-0 - 1)**2 + 1)
        # = -2 - 1 / sqrt(2)
        self.assertAlmostEqual(float(potential[100]), -2.7071068)
        # x[200] = 10 -> p[200] = -2 / sqrt((10 - 0)**2 + 1) - 1 / sqrt((10 - 1)**2 + 1)
        # = -2 / sqrt(101) - 1/ sqrt(82)
        self.assertAlmostEqual(float(potential[200]), -0.30943897)

    def test_get_atomic_chain_potential_exponential_coulomb(self):
        potential = utils.get_atomic_chain_potential(
            grids=jnp.linspace(-10, 10, 201),
            locations=jnp.array([0., 1.]),
            nuclear_charges=jnp.array([2, 1]),
            interaction_fn=utils.exponential_coulomb
        )
        # potential[i] = -sum_a(Z[a] * A * exp(-κ * |x[i] - X[a]|))
        # x[0] = -10 -> p[0] = -A * (2 * exp(-κ * 10) + exp(-κ * 11))
        self.assertAlmostEqual(float(potential[0]), -0.04302428)
        # x[100] = 0 -> p[100] = -A * (2 + exp(-κ))
        self.assertAlmostEqual(float(potential[100]), -2.8470256)
        # x[200] = 10 -> p[200] = -A * (2 * exp(-κ * 10) + exp(-κ * 9))
        self.assertAlmostEqual(float(potential[200]), -0.05699946)

    @parameterized.parameters(
        ([[-0.1], [0.], [0.1]], [1, 3], [1, 2], "grids.ndim is expected to be 1 but got 2"),
        ([-0.1, 0., 0.1], [[1], [3]], [1, 2], "locations.ndim is expected to be 1 but got 2"),
        ([-0.1, 0., 0.1], [1, 3], [[1], [2]], "nuclear_charges.ndim is expected to be 1 but got 2")
    )
    def test_get_atomic_chain_potential_incorrect_ndim(self, grids, locations, nuclear_charges, expected_message):
        with self.assertRaisesRegex(
            ValueError, expected_message
        ):
            utils.get_atomic_chain_potential(
                jnp.array(grids),
                jnp.array(locations),
                jnp.array(nuclear_charges),
                utils.exponential_coulomb
            )

    @parameterized.parameters(
        # 1 / sqrt((1 - 3) ** 2 + 1) * (1 * 2)
        # = 2 / sqrt(5)
        ([1, 3], [1, 2], utils.soft_coulomb, 0.89442719),

        # 1 / sqrt((-2 - 1) ** 2 + 1) * (1 * 1)
        # + 1 / sqrt((-2 - 3) ** 2 + 1) * (1 * 2)
        # + previous
        # = 1 / sqrt(10) + 2 / sqrt(26) + 2 / sqrt(5)
        ([-2, 1, 3], [1, 1, 2], utils.soft_coulomb, 1.602887227),

        # A * exp(-κ*|1 - 3|) * (1 * 2)
        # = 2 * A * exp(-κ*2)
        ([1, 3], [1, 2], utils.exponential_coulomb, 0.92641057),

        # A * exp(-κ|-2 - 1|) * (1 * 1)
        # + A * exp(-κ|-2 - 3|) * (1 * 2)
        # + previous
        # = A * (exp(-κ*3) + 2 * exp(-κ*5) + 2 * exp(-κ*2))
        ([-2, 1, 3], [1, 1, 2], utils.exponential_coulomb, 1.49438414),
    )
    def test_get_nuclear_interaction_energy(
            self, locations, nuclear_charges, interaction_fn, expected_energy):
        self.assertAlmostEqual(
            float(
                utils.get_nuclear_interaction_energy(
                    jnp.array(locations),
                    jnp.array(nuclear_charges),
                    interaction_fn
                )
            ),
            expected_energy
        )


    @parameterized.parameters(
        # 1 / sqrt((1 - 3) ** 2 + 1) * (1 * 2)
        # = 2 / sqrt(5)
        ([1, 3], [1, 2], utils.soft_coulomb, 0.89442719),

        # 1 / sqrt((-2 - 1) ** 2 + 1) * (1 * 1)
        # + 1 / sqrt((-2 - 3) ** 2 + 1) * (1 * 2)
        # + previous
        # = 1 / sqrt(10) + 2 / sqrt(26) + 2 / sqrt(5)
        ([-2, 1, 3], [1, 1, 2], utils.soft_coulomb, 1.602887227),

        # A * exp(-κ*|1 - 3|) * (1 * 2)
        # = 2 * A * exp(-κ*2)
        ([1, 3], [1, 2], utils.exponential_coulomb, 0.92641057),

        # A * exp(-κ|-2 - 1|) * (1 * 1)
        # + A * exp(-κ|-2 - 3|) * (1 * 2)
        # + previous
        # = A * (exp(-κ*3) + 2 * exp(-κ*5) + 2 * exp(-κ*2))
        ([-2, 1, 3], [1, 1, 2], utils.exponential_coulomb, 1.49438414),
    )
    def test_get_nuclear_interaction_energy(
            self, locations, nuclear_charges, interaction_fn, expected_energy):
        self.assertAlmostEqual(
            float(
                utils.get_nuclear_interaction_energy(
                    jnp.array(locations),
                    jnp.array(nuclear_charges),
                    interaction_fn
                )
            ),
            expected_energy
        )

    @parameterized.parameters(
        # For first row in both cases see test_get_nuclear_interaction_energy

        # Second row: 1 / sqrt((0 - 0) ** 2 + 1) * (1 * 1) = 1
        ([[1, 3], [0, 0]], [[1, 2], [1, 1]], utils.soft_coulomb, [0.89442719, 1.]),

        # Second row: A * sqrt(-κ|0 - 0|) * (1 * 1) = A
        ([[1, 3], [0, 0]], [[1, 2], [1, 1]], utils.exponential_coulomb, [0.92641057, 1.071295]),
    )
    def test_get_nuclear_interaction_energy_batch(
            self, locations, nuclear_charges, interaction_fn, expected_energy):
        np.testing.assert_allclose(
            utils.get_nuclear_interaction_energy_batch(
                    jnp.array(locations),
                    jnp.array(nuclear_charges),
                    interaction_fn
                ),
            expected_energy
        )

    @parameterized.parameters(
        ([[1], [3]], [1, 2], "locations.ndim is expected to be 1 but got 2"),
        ([1, 3], [[1], [2]], "nuclear_charges.ndim is expected to be 1 but got 2")
    )
    def test_get_nuclear_interaction_energy_incorrect_ndim(self, locations, nuclear_charges, expected_message):
        with self.assertRaisesRegex(
            ValueError, expected_message
        ):
            utils.get_nuclear_interaction_energy(
                jnp.array(locations),
                jnp.array(nuclear_charges),
                utils.exponential_coulomb
            )

    @parameterized.parameters(-0.1, 0.0, 0.1, 0.2, 0.3)
    def test_float_value_in_array_true(self, value):
        self.assertTrue(
            utils._float_value_in_array(
                jnp.array([-0.1, 0.0, 0.1, 0.2, 0.3]),
                value
            )
        )

    @parameterized.parameters(-0.15, 0.05, 0.12)
    def test_float_value_in_array_false(self, value):
        self.assertFalse(
            utils._float_value_in_array(
                jnp.array([-0.1, 0.0, 0.1, 0.2, 0.3]),
                value
            )
        )

    def test_flip_and_average_the_front_of_array_center_on_grids(self):
        np.testing.assert_allclose(
            utils.flip_and_average(
                locations=jnp.array([-0.1, 0.3]),
                grids=jnp.array([-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8])
            ),
            # centre = 0.1 is in grids
            # centre_index = left_index = right_index = 3
            # len(grids) = 9
            # radius = min(3, 9 - 3 - 1) = min(3, 5) = 3
            # range_slice = slice(3 - 3, 3 + 3 + 1) = slice(0, 7)
            # array_to_flip = [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5]
            # averaging the following where "-" denotes the skipped locations
            #  [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]
            #  [0.5, 0.3, 0.2, 0.7, 0.6, 0.2, 0.1,   -,   -]
            [0.3, 0.25, 0.4, 0.7, 0.4, 0.25, 0.3, 0.1, 0.8]
        )

    def test_flip_and_average_the_back_of_array_center_on_grids(self):
        np.testing.assert_allclose(
            utils.flip_and_average(
                locations=jnp.array([0.4, 0.6]),
                grids=jnp.array([-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]),
            ),
            # centre = 0.5 is in grids
            # centre_index = left_index = right_index = 7
            # len(grids) = 9
            # radius = min(7, 9 - 7 - 1) = min(7, 1) = 1
            # range_slice = slice(-1 + 7, 1 + 7 + 1) = slice(6, 9)
            # array_to_flip = [0.5, 0.1, 0.8]
            # averaging the following where "-" denotes the skipped locations
            #  [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]
            #  [  -,   -,   -,   -,   -,   -, 0.8, 0.1, 0.5]
            [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.65, 0.1, 0.65]
        )

    def test_flip_and_average_the_front_of_array_center_not_on_grids(self):
        np.testing.assert_allclose(
            utils.flip_and_average(
                locations=jnp.array([-0.1, 0.2]),
                grids=jnp.array([-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8])
            ),
            # centre = 0.05 not on grids
            # For left index find index of grid closest to centre but less which is 2 (grid[2] = 0.)
            # For right index find index of grid closest to centre but greater which is 3 (grid[3] = 0.1)

            # In detail:
            # abs_distance_to_centre = [0.25, 0.15, 0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
            # grids < centre = [1, 1, 1, 0, 0, 0, 0, 0, 0]
            # jnp.where([1, 1, 1, 0, 0, 0, 0, 0, 0],
            #                [0.25, 0.15, 0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55], jnp.inf)
            # ) = [0.25, 0.15, 0.05, oo, oo, oo, oo, oo, oo]
            # argmin of above is left_index = 2
            # grids > centre = [0, 0, 0, 1, 1, 1, 1, 1, 1]
            # jnp.where([0, 0, 0, 1, 1, 1, 1, 1, 1],
            #           [0.25, 0.15, 0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55], jnp.inf)
            # ) = [oo, oo, oo, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
            # argmin of above is right_index = 3

            # Flip around [0., 0.1] extending to 2 elements on either side
            # which is the min of the distance to either end of the array
            # from this dual centre
            # In detail:
            # len(grids) = 9
            # radius = min(2, 9 - 3 - 1) = min(2, 5) = 2
            # range_slice = slice(2 - 2, 3 + 2 + 1) = slice(0, 6)
            # array_to_flip = [0.1, 0.2, 0.6, 0.7, 0.2, 0.3]
            # averaging the following where "-" denotes the skipped locations
            #  [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]
            #  [0.3, 0.2, 0.7, 0.6, 0.2, 0.1,  -,   -,   -]
            [0.2, 0.2, 0.65, 0.65, 0.2, 0.2, 0.5, 0.1, 0.8]
        )

    def test_flip_and_average_the_back_of_array_center_not_on_grids(self):
        np.testing.assert_allclose(
            utils.flip_and_average(
                locations=jnp.array([0.4, 0.5]),
                grids=jnp.array([-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8])
            ),
            # centre = 0.45 not on grids
            # index with value closest to centre below is left_index = 6
            # index with value closest to centre above is right_index = 7

            # flip about [0.4, 0.5] extending 1 element on either side
            # [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]
            # [  -,   -,   -,   -,   -, 0.8, 0.1, 0.5, 0.3]
            [0.1, 0.2, 0.6, 0.7, 0.2, 0.55, 0.3, 0.3, 0.55]
        )

    def test_location_centre_at_grids_centre_point_true(self):
        self.assertTrue(
            utils.location_centre_at_grids_centre_point(
                locations=jnp.array([-0.5, 0.5]),
                grids=jnp.array([-0.4, -0.2, 0., 0.2, 0.4])
            )
        )

    def test_location_centre_at_grids_centre_point_false(self):
        self.assertFalse(
            # 0.05 != 0.
            utils.location_centre_at_grids_centre_point(
                locations=jnp.array([-0.5, 0.6]),
                grids=jnp.array([-0.4, -0.2, 0., 0.2, 0.4])
            )
        )

        self.assertFalse(
            # Even elements in grids so no single centre
            utils.location_centre_at_grids_centre_point(
                locations=jnp.array([-0.5, 0.6]),
                grids=jnp.array([-0.4, -0.2, 0., 0.2])
            )
        )

    def test_compute_distances_between_nuclei(self):
        np.testing.assert_allclose(
            utils.compute_distances_between_nuclei(
                locations=np.array([
                    [-1., 1., 3.5, 5.],
                    [-4., 0., 3.5, 10.],
                    [-2., -1., 3.5, 55.],
                ]),
                nuclei_indices=(1, 2)
            )
            [[2.5, 3.5, 4.5]]
        )



