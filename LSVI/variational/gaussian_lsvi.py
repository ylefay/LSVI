from typing import Callable

import jax
import jax.numpy as jnp

from variational.exponential_family import GenericNormalDistribution
from variational.utils import vec, unvec, get_residual

# jax.config.update('jax_platform_name', 'cpu')  # Required to use sqrtm.


def gaussian_lsvi(OP_key: jax.Array, tgt_log_density: Callable, upsilon_init: jnp.ndarray, n_iter: int, n_samples: int,
                  lr_schedule=1.0, return_all=False,
                  target_residual_schedule=jnp.inf):
    """
    Dense Gaussian scheme following Nicolas' note.
    """
    # jax.config.update('jax_platform_name', 'cpu')  # Required to use jax.scipy.linalg.sqrtm.
    dimension = int(jnp.sqrt(len(upsilon_init) - 3 / 4) - 1 / 2)
    normal = GenericNormalDistribution(dimension=dimension)
    sanity = normal.sanity
    statistic = jax.vmap(normal.sufficient_statistic)

    def from_gammatildetilde_to_gammatilde(gammatildetilde):
        gamma2tildetilde = gammatildetilde.at[dimension:int(dimension * (dimension + 1) / 2) + dimension].get()
        gamma1tilde = gammatildetilde.at[:dimension].get()
        gamma0tilde = gammatildetilde.at[-1].get()
        gamma2tilde_matrix = jnp.zeros((dimension, dimension))
        gamma2tilde_matrix = 0.5 * gamma2tilde_matrix.at[jnp.triu_indices(dimension)].set(gamma2tildetilde)
        gamma2tilde_matrix = gamma2tilde_matrix + gamma2tilde_matrix.T
        gamma2tilde = gamma2tilde_matrix.reshape(-1)
        return jnp.concatenate([gamma1tilde, gamma2tilde, jnp.array([gamma0tilde])])

    def from_gammatilde_to_gamma(gammatilde):
        gamma1 = gammatilde.at[:dimension].get()
        gamma2 = gammatilde.at[dimension:dimension ** 2 + dimension].get()
        gamma2_of_interest = gamma2.at[0::(dimension + 1)].get()
        gamma2 = gamma2.at[0::(dimension + 1)].set(gamma2_of_interest
                                                   * 1 / jnp.sqrt(2))
        gamma0 = gammatilde.at[-1].get() - jnp.sum(gamma2_of_interest) * 1 / jnp.sqrt(2)
        return jnp.concatenate([gamma1, gamma2, jnp.array([gamma0])])

    def from_gamma_to_upsilon(current_mean, current_sqrt, gamma):
        inv_chol = jax.scipy.linalg.inv(current_sqrt)  # O(n^3/2)
        gamma2 = gamma.at[dimension:dimension ** 2 + dimension].get()
        gamma1 = gamma.at[:dimension].get()
        gamma0 = gamma.at[-1].get()
        B = unvec(gamma2, shape=(dimension, dimension))
        upsilon2 = vec(inv_chol @ B @ inv_chol.T)  # O(n^3) equal to jnp.kron(inv_chol, inv_chol)@gamma2
        upsilon1 = ((gamma1.T - 2 * upsilon2.T @ (jnp.kron(current_mean[:, jnp.newaxis], current_sqrt))) @ inv_chol).T
        upsilon0 = gamma0 - upsilon1.T @ current_mean - upsilon2.T @ vec(
            current_mean[:, jnp.newaxis] @ current_mean[:, jnp.newaxis].T)
        upsilon = jnp.concatenate([upsilon1, upsilon2, jnp.array([upsilon0])])
        return upsilon

    def from_gammatildetilde_to_gamma(gammatildetilde):
        return from_gammatilde_to_gamma(from_gammatildetilde_to_gammatilde(gammatildetilde))

    @jax.vmap
    def modified_statistic(z):
        vecZZt = vec(z[:, jnp.newaxis] @ z[:, jnp.newaxis].T)
        vecZZt = vecZZt.at[0::(dimension + 1)].set((vecZZt.at[0::(dimension + 1)].get() - 1) / jnp.sqrt(2))
        vectriuunvecvecZZt = vec(unvec(vecZZt, (dimension, dimension)).at[jnp.triu_indices(dimension)].get())
        return jnp.concatenate([z, vectriuunvecvecZZt, jnp.array([1.])])

    def momentum_backtracking(lr, upsilon, next_upsilon, y, X, target_residual):
        lr = jax.lax.while_loop(lambda _lr: sanity(next_upsilon * _lr + (1 - _lr) * upsilon),
                                lambda _lr: _lr / 2, lr)
        current_residual = get_residual(y, X, next_upsilon * lr + (1 - lr) * upsilon)
        lr_tempering = jax.lax.cond(current_residual <= target_residual, lambda _: lr,
                                    lambda _: jnp.sqrt(target_residual / current_residual), None)
        lr = jax.lax.min(lr, lr_tempering)
        new_residual = get_residual(y, X, next_upsilon * lr + (1 - lr) * upsilon)
        return lr, new_residual

    def iter_routine(upsilon, inps):
        key, lr, target_residual = inps
        theta = upsilon.at[:-1].get()
        current_mean, current_cov = normal.get_mean_cov(theta)
        # sqrtm = jnp.real(jax.scipy.linalg.sqrtm(current_cov))  # seems more stable, but CPU only compatible.
        # sqrtm = jax.scipy.linalg.cholesky(current_cov) # numerical issues
        # sqrtm = jax.scipy.linalg.cholesky((current_cov + current_cov.T) / 2)  # seems more stable and GPU compatible
        D, V = jax.scipy.linalg.eigh(current_cov)
        sqrtm = (V * jnp.sqrt(D)) @ V.T
        samples = sampling(key)
        y = vmapped_tgt_log_density(current_mean[jnp.newaxis, :] + samples @ sqrtm)
        X = modified_statistic(samples)
        next_gamma_tilde_tilde = X.T @ y / n_samples  # OLS(X, y) works well..
        next_gamma = from_gammatildetilde_to_gamma(next_gamma_tilde_tilde)
        next_upsilon = from_gamma_to_upsilon(current_mean, sqrtm, next_gamma)
        lr, residual = momentum_backtracking(lr, upsilon, next_upsilon, y, statistic(samples), target_residual)
        next_upsilon = next_upsilon * lr + (1 - lr) * upsilon
        return next_upsilon, next_gamma_tilde_tilde, residual

    def fun_iter(upsilon, inps):
        next_upsilon, *_ = iter_routine(upsilon, inps)
        return next_upsilon, next_upsilon

    def fun_iter_return_all(upsilon, inps):
        next_upsilon, next_gamma_tilde_tilde, residual = iter_routine(upsilon, inps)
        return next_upsilon, (next_upsilon, next_gamma_tilde_tilde, residual)

    sampling = lambda keys: jax.random.normal(keys, shape=(n_samples, dimension))
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)
    iter_keys = jax.random.split(OP_key, n_iter)

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    if isinstance(target_residual_schedule, float):
        target_residual_schedule = jnp.full(n_iter, target_residual_schedule)

    if return_all:
        _, all_results = jax.lax.scan(fun_iter_return_all, upsilon_init,
                                      (iter_keys, lr_schedule, target_residual_schedule))
        upsilons = all_results[0]
        upsilons = jnp.insert(upsilons, 0, upsilon_init, axis=0)
        return upsilons, all_results[1:]
    else:
        _, upsilons = jax.lax.scan(fun_iter, upsilon_init, (iter_keys, lr_schedule, target_residual_schedule))
        upsilons = jnp.insert(upsilons, 0, upsilon_init, axis=0)
        return upsilons, None
