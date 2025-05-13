from typing import Callable

import jax
import jax.numpy as jnp

from variational.utils import OLS, get_residual


def lsvi(OP_key: jax.Array, sampling: Callable, sufficient_statistic: Callable, tgt_log_density: Callable,
         upsilon_init: jnp.ndarray, n_iter: int, n_samples: int,
         regression=OLS, lr_schedule=1.0, return_all=False, sanity=lambda _: False, target_residual_schedule=jnp.inf):
    """
    Fixed-point scheme for Variational Inference problem on exponential families, given some regression estimators.
    :param OP_key: PRNGKey, needed to generate samples from both the target and current fitted distribution
    :param sampling: sampling method from the variational family
    :param sufficient_statistic: sufficient statistic of the variational family
    :param tgt_log_density: log-density of the target distribution
    :param upsilon_init: initial parameter characterizing the initial variational distribution
    :param n_iter: number of iterations of the fixed-point scheme
    :param n_samples: number of samples to draw at each iteration, used to replace the exact expectations by empirical
        expectations
    :param regression: regression estimator, typically OLS.
    :param lr_schedule: float or array of floats, learning rate schedule
    :param return_all: bool, whether to return all the intermediate results, only the residual variances //including samples and evaluation of log-density
    :param sanity: callable, function to check whether a natural parameter defines a valid distribution, if set then call momentum_backtracking
    :param target_residual_schedule: float or array of floats, desired variance for the residuals
    """

    vmapped_sampling = jax.vmap(sampling, in_axes=(None, 0))
    vmapped_sufficient_statistic = jax.vmap(sufficient_statistic)
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)
    iter_keys = jax.random.split(OP_key, n_iter)

    def momentum_backtracking(lr, upsilon, next_upsilon, y, X, target_residual):
        """
        Momentum backtracking to ensure that the natural parameter defines a valid distribution
        This function divides by two the learning rate until the natural parameter defines a valid distribution
        Then compare the obtained residual variance with the target residual variance and compute lr_tempering such that
        the new residuals have variance than the target. Take the minimum between the two learning_rate.
        """
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
        keys = jax.random.split(key, n_samples)
        theta = upsilon.at[:-1].get()
        samples = vmapped_sampling(theta, keys)
        X = vmapped_sufficient_statistic(samples)
        y = vmapped_tgt_log_density(samples)
        next_upsilon = regression(X, y)
        lr, residual = momentum_backtracking(lr, upsilon, next_upsilon, y, X, target_residual)
        next_upsilon = next_upsilon * lr + (1 - lr) * upsilon
        return next_upsilon, residual

    def fun_iter(upsilon, inps):
        next_upsilon, _ = routine_iter(upsilon, inps)
        return next_upsilon, next_upsilon

    def fun_iter_return_all(upsilon, inps):
        next_upsilon, residual = routine_iter(upsilon, inps)
        return next_upsilon, (next_upsilon, residual)

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
