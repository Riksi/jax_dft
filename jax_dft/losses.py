"""Loss functions for optimization."""

import functools

import jax
import jax.numpy as jnp


def mean_square_error(target, predict):
    return jnp.mean((target - predict) ** 2)


def _get_discount_coefficients(num_steps, discount):
    # last index is num_steps - 1
    # coef[i] = discount ** (num_steps - 1 - i)
    # ----> discount  ** (num_steps - 1 - i), ...  discount, 1.
    # e.g. for num_steps=7, discount = 0.9 you get
    # [0.531441, 0.59049, 0.6561, 0.729, 0.81, 0.9, 1]

    return jnp.power(discount, jnp.arange(num_steps - 1, -1, -1))


@functools.partial(jax.jit, static_argnums=(1,))
def _trajectory_error(error, discount):
    batch_size = error.shape[0]
    num_steps = error.shape[1]
    # [B, N]
    error = jnp.mean(
        error.reshape((batch_size, num_steps, -1)), axis=2)
    # dot([B, N], [N]) -> [B]
    discounted_mse = jnp.dot(error, _get_discount_coefficients(num_steps, discount))
    return jnp.mean(discounted_mse)


def trajectory_error(error, discount):
    return _trajectory_error(error, discount)


@functools.partial(jax.jit, static_argnums=(2,))
def _trajectory_mse(target, predict, discount):
    # predict: [d0, num_steps, d2, ..., d_{n-1}]
    # target: [d0, d2, ..., d_{n-2}]
    if predict.ndim < 2:
        raise ValueError(
            f'predict should have ndim >= 2 got {predict.ndim}'
        )
    if predict.ndim - target.ndim != 1:
        raise ValueError(
            f'predict should have ndim that is greater than target ndim by 1'
            f' got predict.ndim={predict.ndim}, target.ndim={target.ndim}'
        )
    # [d0, d2, ..., d_{n-2}]- > [d0, 1, d2, ..., d_{n-1}]
    target = jnp.expand_dims(target, axis=1)
    return trajectory_error((target - predict) ** 2, discount)


def trajectory_mse(target, predict, discount):
    return _trajectory_mse(target, predict, discount)