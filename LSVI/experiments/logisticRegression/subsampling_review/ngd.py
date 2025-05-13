import pickle
from typing import Callable

import jax
import jax.numpy as jnp

from experiments.logisticRegression.subsampling_review.get_dataset import get_Census_Income_dataset
from experiments.logisticRegression.subsampling_review.utils import get_tgt_log_density
from variational.exponential_family import GenericMeanFieldNormalDistribution, MeanFieldNormalDistribution

OUTPUT_PATH = "./output_mean_field"
jax.config.update("jax_enable_x64", True)


def ngd_on_mf_gaussian_kl(OP_key: jax.Array, tgt_log_density: Callable,
                          upsilon_init: jnp.ndarray, n_iter: int, n_samples: int,
                          lr_schedule=1.0, sanity=lambda _: False):
    r"""
    Natural gradient descent algorithm for variational inference within exponential families
    NOTE : the objective is the MC version of \int \bar{q}(\log(\bar{q} / \pi) = KL(\bar{q}\mid \bar{\pi}) up to some additive constant
    :param OP_key: PRNGKey, needed to generate samples from both the target and current fitted distribution
    :param tgt_log_density: log-density of the target distribution
    :param upsilon_init: initial parameter characterizing the initial variational distribution
    :param n_iter: number of iterations of the fixed-point scheme
    :param n_samples: number of samples to draw at each iteration, used to replace the exact expectations by empirical
        expectations
    :param lr_schedule: float or array of floats, learning rate schedule
    :param sanity: callable, function to check whether a natural parameter defines a valid distribution, if set then call momentum_backtracking
    """
    dimension = int((len(upsilon_init) - 1) / 2)
    gaussian = GenericMeanFieldNormalDistribution(dimension)

    def make_logpdf(upsilon):
        def logpdf(x):
            mean, cov = gaussian.get_mean_cov(upsilon.at[:-1].get())
            L = cov ** 0.5
            y = (x - mean) / L
            return (-1 / 2 * jnp.einsum('...i,...i->...', y, y) - dimension / 2 * jnp.log(2 * jnp.pi)
                    - jnp.log(L).sum(-1))

        return logpdf

    sampling = gaussian.sampling_method
    sufficient_statistic = gaussian.sufficient_statistic
    vmapped_sampling = jax.vmap(sampling, in_axes=(None, 0))
    vmapped_sufficient_statistic = jax.vmap(sufficient_statistic)
    vmapped_tgt_log_density = jax.vmap(tgt_log_density, in_axes=(None, 0))
    iter_keys = jax.random.split(OP_key, n_iter)

    def momentum_backtracking(lr, upsilon, next_upsilon):
        """
        Momentum backtracking to ensure that the natural parameter defines a valid distribution
        This function divides by two the learning rate until the natural parameter defines a valid distribution
        """
        lr = jax.lax.while_loop(lambda _lr: sanity(- next_upsilon * _lr + upsilon),
                                lambda _lr: _lr / 2, lr)
        return lr

    def kl(upsilon, keys):
        theta = upsilon.at[:-1].get()
        samples = vmapped_sampling(theta, keys)
        logq = make_logpdf(upsilon)(samples)
        integrand = logq - vmapped_tgt_log_density(keys.at[0].get(), samples)
        estimate_of_kl = jnp.sum(integrand)
        return estimate_of_kl, samples

    def routine_iter(upsilon, inps):
        key, lr = inps
        keys = jax.random.split(key, n_samples)
        val, grad = jax.value_and_grad(kl, has_aux=True)(upsilon, keys)
        _, samples = val
        X = vmapped_sufficient_statistic(samples)
        next_upsilon = jnp.linalg.pinv(X.T @ X) @ grad
        lr = momentum_backtracking(lr, upsilon, next_upsilon)
        next_upsilon = upsilon - lr * next_upsilon
        return next_upsilon

    def fun_iter(upsilon, inps):
        next_upsilon = routine_iter(upsilon, inps)
        return next_upsilon, next_upsilon

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    _, upsilons = jax.lax.scan(fun_iter, upsilon_init, (iter_keys, lr_schedule))
    upsilons = jnp.insert(upsilons, 0, upsilon_init, axis=0)
    return upsilons


def experiment(keys, n_samples=100000, n_iter=100, lr_schedule=None,
               OUTPUT_PATH="./output_mean_field", title=""):
    flipped_predictors = jnp.array(get_Census_Income_dataset())
    N, dim = flipped_predictors.shape
    P = 1000

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_covariance = jnp.diag(my_prior_covariance)
    my_prior_log_density = MeanFieldNormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(P, flipped_predictors, my_prior_log_density)

    # Mean Field Gaussian Variational Family
    my_variational_family = GenericMeanFieldNormalDistribution(dimension=dim)
    sanity = my_variational_family.sanity
    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim))

    @jax.vmap
    def f(key):
        res = ngd_on_mf_gaussian_kl(key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                    lr_schedule=lr_schedule, sanity=sanity)
        return res

    res = f(keys)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule}
    desc = "Census dataset, mean-field Gaussian, NGD"
    with open(
            f"{OUTPUT_PATH}/gaussian_meanfield_ngd_{n_iter}_{n_samples}_{title}.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': None}, f)


if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)
    n_iter = 1000
    Seq_titles = ['Seq1', 'Seq1em3']
    Seq = [jnp.ones(n_iter), jnp.ones(n_iter) * 1e-3]
    Ns = [1e4]
    n_repetitions = 100
    keys = jax.random.split(OP_key, n_repetitions)
    for idx, title in enumerate(Seq_titles):
        print(title)
        for n_samples in Ns:
            experiment(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq[idx],
                       OUTPUT_PATH=OUTPUT_PATH, title=title)
