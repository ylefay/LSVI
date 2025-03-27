import os.path
import pickle

from typing import Callable

import jax
import jax.numpy as jnp

from variational.utils import get_residual
from experiments.logisticRegression.subsampling_review.utils import get_tgt_log_density
from experiments.logisticRegression.subsampling_review.get_dataset import get_Census_Income_dataset
from variational.exponential_family import GenericMeanFieldNormalDistribution, NormalDistribution

OUTPUT_PATH = "./output_mean_field"
OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def mean_field_gaussian_lsvi(OP_key: jax.Array, tgt_log_density: Callable, upsilon_init: jnp.ndarray, n_iter: int,
                             n_samples: int, lr_schedule=1.0,
                             return_all=False,
                             target_residual_schedule=jnp.inf):
    """
    Mean-field Gaussian scheme following Nicolas' note.
    replace the log-density by the sub-sampling estimate
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
        y = vmapped_tgt_log_density(key, current_mean + jnp.sqrt(current_vec_diag_cov) * samples)
        X = modified_statistic(samples)
        next_gamma = X.T @ y / n_samples
        next_upsilon = from_gamma_to_upsilon(current_mean, current_vec_diag_cov, next_gamma)
        lr, residual = momentum_backtracking(lr, upsilon, next_upsilon, y, statistic(samples), target_residual)
        next_upsilon = next_upsilon * lr + (1 - lr) * upsilon
        return next_upsilon, next_gamma, residual

    sampling = lambda keys: jax.random.normal(keys, shape=(n_samples, dimension))
    vmapped_tgt_log_density = jax.vmap(tgt_log_density, in_axes=(None, 0))

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


def experiment(keys, n_samples=100000, n_iter=100, lr_schedule=None, target_residual_schedule=None, title_seq="Seq",
               OUTPUT_PATH="./output"):
    flipped_predictors = jnp.array(get_Census_Income_dataset())
    N, dim = flipped_predictors.shape
    P = 1000
    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(P, flipped_predictors, my_prior_log_density)

    # Mean Field Gaussian Variational Family
    my_variational_family = GenericMeanFieldNormalDistribution(dimension=dim)

    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim))

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule, 'residual': target_residual_schedule}
    desc = "PIMA dataset, heuristic, mf. Gaussian, Nicolas"

    # if not os.path.exists(
    #        f"{OUTPUT_PATH}/heuristic_gaussian_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl"):

    @jax.vmap
    def f(key):
        res, res_all = mean_field_gaussian_lsvi(key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                                lr_schedule=lr_schedule,
                                                target_residual_schedule=target_residual_schedule,
                                                return_all=False)
        return res, res_all

    res, res_all = f(keys)
    with open(
            f"{OUTPUT_PATH}/heuristic_gaussian_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)


if __name__ == "__main__":
    n_iter = 1000
    Seq_titles = ['Seq1u1']
    interval = jnp.arange(1, n_iter + 1)
    Seq = [jnp.ones(n_iter)]
    Ns = [1e4]
    target_residual_schedules = [jnp.full(n_iter, 1)]
    n_repetitions = 100
    keys = jax.random.split(OP_key, n_repetitions)
    for idx, title in enumerate(Seq_titles):
        print(title)
        for n_samples in Ns:
            experiment(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq[idx],
                       target_residual_schedule=target_residual_schedules[idx], title_seq=title,
                       OUTPUT_PATH=OUTPUT_PATH)
