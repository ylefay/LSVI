from typing import Callable

import jax
import jax.numpy as jnp

from variational.exponential_family import GenericMeanFieldNormalDistribution
from variational.utils import get_residual


def mean_field_gaussian_lsvi(OP_key: jax.Array, tgt_log_density: Callable, upsilon_init: jnp.ndarray, n_iter: int,
                             n_samples: int, lr_schedule=1.0,
                             return_all=False,
                             target_residual_schedule=jnp.inf):
    """
    Mean-field Gaussian scheme following Nicolas' note.
    """
    dimension = int((len(upsilon_init) - 1) / 2)
    normal = GenericMeanFieldNormalDistribution(dimension=dimension)
    sanity = normal.sanity
    statistic = jax.vmap(normal.sufficient_statistic)

    def from_gamma_to_upsilon(current_mean, current_vec_diag_cov, gamma):
        gamma2 = gamma.at[dimension:2 * dimension].get()
        gamma1 = gamma.at[:dimension].get()
        gamma0 = gamma.at[-1].get()
        upsilon2 = gamma2 * 1 / current_vec_diag_cov * 1 / jnp.sqrt(2)
        upsilon1 = gamma1 * (1 / jnp.sqrt(current_vec_diag_cov)) - 2 * upsilon2 * current_mean
        upsilon0 = gamma0 - upsilon1.T @ current_mean - upsilon2.T @ (current_mean ** 2 + current_vec_diag_cov)
        upsilon = jnp.concatenate([upsilon1, upsilon2, jnp.array([upsilon0])])
        return upsilon

    @jax.vmap
    def modified_statistic(z):
        return jnp.concatenate([z, (z ** 2 - 1) / jnp.sqrt(2), jnp.array([1.])])

    def momentum_backtracking(lr, upsilon, next_upsilon, y, X, target_residual):
        lr = jax.lax.while_loop(lambda _lr: sanity(next_upsilon * _lr + (1 - _lr) * upsilon),
                                lambda _lr: _lr / 2, lr)
        current_residual = get_residual(y, X, next_upsilon * lr + (1 - lr) * upsilon)
        lr_tempering = jax.lax.cond(current_residual <= target_residual, lambda _: lr,
                                    lambda _: jnp.sqrt(target_residual / current_residual), None)
        lr = jax.lax.min(lr, lr_tempering)
        new_residual = get_residual(y, X, next_upsilon * lr + (1 - lr) * upsilon)
        return lr, new_residual

    def routine_iter(upsilon, inps):
        key, lr, target_residual = inps
        theta = upsilon.at[:-1].get()
        current_mean, current_vec_diag_cov = normal.get_mean_cov(theta)
        samples = sampling(key)
        y = vmapped_tgt_log_density(current_mean + jnp.sqrt(current_vec_diag_cov) * samples)
        X = modified_statistic(samples)
        next_gamma = X.T @ y / n_samples
        next_upsilon = from_gamma_to_upsilon(current_mean, current_vec_diag_cov, next_gamma)
        lr, residual = momentum_backtracking(lr, upsilon, next_upsilon, y, statistic(samples), target_residual)
        next_upsilon = next_upsilon * lr + (1 - lr) * upsilon
        return next_upsilon, next_gamma, residual

    sampling = lambda keys: jax.random.normal(keys, shape=(n_samples, dimension))
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)

    def iter_fun(upsilon, inps):
        next_upsilon, *_ = routine_iter(upsilon, inps)
        return next_upsilon, next_upsilon

    def iter_return_all_fun(upsilon, inps):
        next_upsilon, next_gamma, residual = routine_iter(upsilon, inps)
        return next_upsilon, (next_upsilon, next_gamma, residual)

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    if isinstance(target_residual_schedule, float):
        target_residual_schedule = jnp.full(n_iter, target_residual_schedule)

    iter_keys = jax.random.split(OP_key, n_iter)

    if return_all:
        _, all_results = jax.lax.scan(iter_return_all_fun, upsilon_init,
                                      (iter_keys, lr_schedule, target_residual_schedule))
        upsilons = all_results[0]
        upsilons = jnp.insert(upsilons, 0, upsilon_init, axis=0)
        return upsilons, all_results[1:]
    else:
        _, upsilons = jax.lax.scan(iter_fun, upsilon_init, (iter_keys, lr_schedule, target_residual_schedule))
        upsilons = jnp.insert(upsilons, 0, upsilon_init, axis=0)
        return upsilons, None
