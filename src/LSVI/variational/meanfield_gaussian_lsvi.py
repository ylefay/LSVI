from typing import Callable

import jax
import jax.numpy as jnp

from LSVI.variational.exponential_family import GenericMeanFieldNormalDistribution
from LSVI.variational.utils import get_residual


def mean_field_gaussian_lsvi(OP_key: jax.Array, tgt_log_density: Callable, eta_init: jnp.ndarray, n_iter: int,
                             n_samples: int, lr_schedule=1.0,
                             return_all=False,
                             target_residual_schedule=jnp.inf):
    """
    Mean-field Gaussian scheme.
    See Section 4. of https://arxiv.org/abs/2502.18475,
    Least Squares Variational Inference, Le Fay Y., Chopin N. Barthelmé S. 2025.

    The following computations can be rederived by checking carefully the proofs in appendix Section D.3.
    """
    dimension = int((len(eta_init) - 1) / 2)
    normal = GenericMeanFieldNormalDistribution(dimension=dimension)
    sanity = normal.sanity
    statistic = jax.vmap(normal.sufficient_statistic)

    def from_gamma_to_eta(current_mean, current_vec_diag_cov, gamma):
        gamma2 = gamma.at[dimension:2 * dimension].get()
        gamma1 = gamma.at[:dimension].get()
        gamma0 = gamma.at[-1].get()
        eta2 = gamma2 * 1 / current_vec_diag_cov * 1 / jnp.sqrt(2)
        eta1 = gamma1 * (1 / jnp.sqrt(current_vec_diag_cov)) - 2 * eta2 * current_mean
        eta0 = gamma0 - eta1.T @ current_mean - eta2.T @ (current_mean ** 2 + current_vec_diag_cov)
        eta = jnp.concatenate([eta1, eta2, jnp.array([eta0])])
        return eta

    @jax.vmap
    def modified_statistic(z):
        return jnp.concatenate([z, (z ** 2 - 1) / jnp.sqrt(2), jnp.array([1.])])

    def momentum_backtracking(lr, eta, next_eta, y, X, target_residual):
        lr = jax.lax.while_loop(lambda _lr: sanity(next_eta * _lr + (1 - _lr) * eta),
                                lambda _lr: _lr / 2, lr)
        current_residual = get_residual(y, X, next_eta * lr + (1 - lr) * eta)
        lr_tempering = jax.lax.cond(current_residual <= target_residual, lambda _: lr,
                                    lambda _: jnp.sqrt(target_residual / current_residual), None)
        lr = jax.lax.min(lr, lr_tempering)
        new_residual = get_residual(y, X, next_eta * lr + (1 - lr) * eta)
        return lr, new_residual

    def routine_iter(eta, inps):
        key, lr, target_residual = inps
        theta = eta.at[:-1].get()
        current_mean, current_vec_diag_cov = normal.get_mean_cov(theta)
        samples = sampling(key)
        y = vmapped_tgt_log_density(current_mean + jnp.sqrt(current_vec_diag_cov) * samples)
        X = modified_statistic(samples)
        next_gamma = X.T @ y / n_samples
        next_eta = from_gamma_to_eta(current_mean, current_vec_diag_cov, next_gamma)
        lr, residual = momentum_backtracking(lr, eta, next_eta, y, statistic(samples), target_residual)
        next_eta = next_eta * lr + (1 - lr) * eta
        return next_eta, next_gamma, residual

    sampling = lambda keys: jax.random.normal(keys, shape=(n_samples, dimension))
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)

    def iter_fun(eta, inps):
        next_eta, *_ = routine_iter(eta, inps)
        return next_eta, next_eta

    def iter_return_all_fun(eta, inps):
        next_eta, next_gamma, residual = routine_iter(eta, inps)
        return next_eta, (next_eta, next_gamma, residual)

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    if isinstance(target_residual_schedule, float):
        target_residual_schedule = jnp.full(n_iter, target_residual_schedule)

    iter_keys = jax.random.split(OP_key, n_iter)

    if return_all:
        _, all_results = jax.lax.scan(iter_return_all_fun, eta_init,
                                      (iter_keys, lr_schedule, target_residual_schedule))
        etas = all_results[0]
        etas = jnp.insert(etas, 0, eta_init, axis=0)
        return etas, all_results[1:]
    else:
        _, etas = jax.lax.scan(iter_fun, eta_init, (iter_keys, lr_schedule, target_residual_schedule))
        etas = jnp.insert(etas, 0, eta_init, axis=0)
        return etas, None
