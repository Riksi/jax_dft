"""Tests for jax_dft.losses."""

from absl.testing import absltest
from jax.config import config
import jax.numpy as jnp
import numpy as np

from jax_dft import losses


# Set the default dtype as float64
config.update('jax_enable_x64', True)


class LossesTest(absltest.TestCase):

    def test_trajectory_mse_wrong_predict_ndim(self):
        with self.assertRaisesRegex(
            ValueError,
            'predict should have ndim >= 2 got 1'
        ):
            losses.trajectory_mse(
                target=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
                predict=jnp.array([0.6, 0.6, 0.6, 0.6]),
                discount=1.
            )

    def test_trajectory_mse_wrong_predict_target_ndim_difference(self):
        with self.assertRaisesRegex(
            ValueError,
            ('predict should have ndim that is greater than target ndim by 1'
             ' got predict.ndim=2, target.ndim=2')
        ):
            losses.trajectory_mse(
                target=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
                predict=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
                discount=1.
            )

    def test_density_mse(self):
        self.assertAlmostEqual(
            float(losses.mean_square_error(
                target=jnp.array([[0.2, 0.2, 0.2, 0.2],
                                  [0.6, 0.6, 0.6, 0.6]]),
                predict=jnp.array([[0.4, 0.5, 0.2, 0.3],
                                   [0.6, 0.6, 0.6, 0.6]])
            )),
            # (0.2**2 + 0.3**2 + 0.1**2) / 8
            0.0175
        )

    def test_energy_mse(self):
        self.assertAlmostEqual(
            float(losses.mean_square_error(
                target=jnp.array([[0.2, 0.6]]),
                predict=jnp.array([[0.4, 0.7]])
            )),
            # (0.2**2 + 0.1**2) / 2
            0.025
        )

    def test_get_discount_coefficients(self):
        np.testing.assert_allclose(
            losses._get_discount_coefficients(
                4, 0.8
            ), [0.512, 0.64, 0.8, 1.]
        )

    def test_trajectory_mse_on_density(self):
        self.assertAlmostEqual(
            float(
                losses.trajectory_mse(
                    target=jnp.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]]),
                    predict=jnp.array([
                        [[0.4, 0.5, 0.2, 0.3],
                         [0.3, 0.3, 0.2, 0.2],
                         [0.3, 0.3, 0.3, 0.2]],
                        [[0.6, 0.6, 0.6, 0.6],
                         [0.6, 0.6, 0.6, 0.5],
                         [0.6, 0.6, 0.6, 0.6]]]),
                    discount=0.6
                )
            ),
            # (
            #   (
            #       (0.2**2 + 0.3**2 + 0.1**2) / 4 * 0.36
            #     + (0.1**2 + 0.1**2) / 4 * 0.6
            #     + (0.1**2 + 0.1**2 + 0.1**2) / 4 * 1
            #   )
            # + (
            #       0.
            #     + (0.1**2) / 4 * 0.6
            #     + 0.
            #    )
            # ) / 2
            0.0123
        )

    def test_trajectory_mse_on_energy(self):
        self.assertAlmostEqual(
            float(
                losses.trajectory_mse(
                    target=jnp.array([0.2, 0.6]),
                    predict=jnp.array([[0.4, 0.3, 0.2], [0.7, 0.7, 0.7]]),
                    discount=0.6
                )
            ),
            # ((0.2**2 * 0.36 + 0.1**2 * 0.6)
            #   + (0.1**2 * 0.36 + 0.1**2 * 0.6 + 0.1**2 * 1.))/2
            0.02
        )


if __name__ == '__main__':
    absltest.main()

