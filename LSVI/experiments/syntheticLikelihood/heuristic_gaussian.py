import os.path
import pickle

import jax
import jax.numpy as jnp

from experiments.syntheticLikelihood.fowler_toad import get_tgt_density
from variational.exponential_family import GenericNormalDistribution
from variational.utils import OLS, get_residual


def lsvi(OP_key, sampling, sufficient_statistic, tgt_log_density, upsilon_init, n_iter, n_samples,
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
        y = tgt_log_density(key, samples)
        next_upsilon = regression(X, y)
        lr, residual = momentum_backtracking(lr, upsilon, next_upsilon, y, X, target_residual)
        next_upsilon = next_upsilon * lr + (1 - lr) * upsilon
        return next_upsilon, residual

    def iter(upsilon, inps):
        next_upsilon, _ = routine_iter(upsilon, inps)
        return next_upsilon, next_upsilon

    def iter_return_all(upsilon, inps):
        next_upsilon, residual = routine_iter(upsilon, inps)
        return next_upsilon, (next_upsilon, residual)

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    if isinstance(target_residual_schedule, float):
        target_residual_schedule = jnp.full(n_iter, target_residual_schedule)

    if return_all:
        _, all_results = jax.lax.scan(iter_return_all, upsilon_init, (iter_keys, lr_schedule, target_residual_schedule))
        upsilons = all_results[0]
        upsilons = jnp.insert(upsilons, 0, upsilon_init, axis=0)
        return upsilons, all_results[1:]
    else:
        _, upsilons = jax.lax.scan(iter, upsilon_init, (iter_keys, lr_schedule, target_residual_schedule))
        upsilons = jnp.insert(upsilons, 0, upsilon_init, axis=0)
        return upsilons, None


def experiment(n_samples=100000, n_iter=100, lr_schedule=None, target_residual_schedule=None, title_seq="Seq",
               OP_key=jax.random.PRNGKey(0), OUTPUT_PATH="./output"):
    sdata = pickle.load(open(f"{OUTPUT_PATH}/ht_data_63_66_[0 0].pkl", "rb"))

    scales2 = jnp.array([1., 1., 1.])
    dim = 3
    tgt_log_density = get_tgt_density(sdata, 100, shrinkage=0.5, transform=True, scales2=scales2)
    my_variational_family = GenericNormalDistribution(dimension=dim)

    scales = jnp.array([1., 100., 0.9])
    init_mean = jnp.array([1.5, 50, 0.5])
    init_mean = (init_mean - jnp.array([1., 0., 0.])) / scales
    init_mean = jax.scipy.special.logit(init_mean) / scales2
    init_cov = jnp.diag(jnp.array([0.1, 0.1, 0.1]))
    upsilon_init = my_variational_family.get_upsilon(init_mean, init_cov)

    sampling = my_variational_family.sampling_method
    sufficient_statistic = my_variational_family.sufficient_statistic
    sanity = my_variational_family.sanity

    if lr_schedule is None:
        lr_schedule = 1 / jnp.arange(1, n_iter + 1)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule, 'target_residual': target_residual_schedule}
    desc = "Synthetic Likelihood experiment, Fowler's toad, Heuristic"
    if not os.path.exists(
            f"{OUTPUT_PATH}/heuristic_gaussian_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl"):
        res, res_all = lsvi(OP_key, sampling, sufficient_statistic, tgt_log_density, upsilon_init, n_iter, n_samples,
                            lr_schedule=lr_schedule, sanity=sanity,
                            target_residual_schedule=target_residual_schedule,
                            return_all=False)
        with open(
                f"{OUTPUT_PATH}/heuristic_gaussian_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl",
                "wb") as f:
            pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    OP_key = jax.random.PRNGKey(4)
    Seq_titles = ['inv_u_1_shrinkage_05']
    OUTPUT_PATH = "./output"
    n_iter = 50
    interval = jnp.arange(1, n_iter + 1)
    Seq = [1 / interval]
    Ns = [100]
    target_residual_schedule = jnp.inf

    for idx, title in enumerate(Seq_titles):
        print(title)
        for n_samples in Ns:
            for key in range(1):
                print(key)
                print(n_samples)
                experiment(n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq[idx], title_seq=title,
                           target_residual_schedule=target_residual_schedule,
                           OP_key=jax.random.PRNGKey(key))
