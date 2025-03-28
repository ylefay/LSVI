import logging

import jax
import jax.numpy as jnp
import jax.random
import optax
from blackjax.vi import meanfield_vi
from timeit_decorator import timeit

from experiments.logisticRegression.mnist.load_mnist import mnist_dataset
from experiments.logisticRegression.utils import get_tgt_log_density
from variational.exponential_family import GenericMeanFieldNormalDistribution, NormalDistribution
from variational.exponential_family import MeanFieldNormalDistribution
from variational.meanfield_gaussian_lsvi import mean_field_gaussian_lsvi

n_runs = 5

OP_key = jax.random.PRNGKey(0)

jax.config.update("jax_enable_x64", True)


@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def experiment_mf_lsvi(key, n_samples, n_iter, lr_schedule=None, target_residual_schedule=None):
    flipped_predictors = mnist_dataset(return_test=False)
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Mean Field Gaussian Variational Family
    my_variational_family = GenericMeanFieldNormalDistribution(dimension=dim)

    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim) * jnp.exp(-2))

    def f(key):
        res, res_all = mean_field_gaussian_lsvi(key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                                lr_schedule=lr_schedule,
                                                target_residual_schedule=target_residual_schedule,
                                                return_all=False)
        return res, res_all

    _, _ = f(key)
    return None


@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def experiment(key, num_iter, num_samples, sgd=1e-3):
    flipped_predictors = mnist_dataset(return_test=False)
    dim = flipped_predictors.shape[1]

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = jnp.diag(my_prior_covariance)
    my_prior_log_density = MeanFieldNormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)
    opt = optax.sgd(sgd)
    res = meanfield_vi.as_top_level_api(tgt_log_density, optimizer=opt, num_samples=num_samples)
    initial_state = res.init(position=jnp.zeros(dim))

    def inference_loop(rng_key):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = res.step(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_iter)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        mus = jnp.array(states.mu)
        mus = jnp.insert(mus, 0, jnp.array(initial_state.mu), 0)
        rhos = jnp.array(states.rho)
        rhos = jnp.insert(rhos, 0, jnp.array(initial_state.rho), 0)

        return mus, rhos

    states = inference_loop(key)
    return None


if __name__ == "__main__":
    """
    Running n_runs time with timeit decorator the experiment mean field lsvi for different n_samples
    """
    Seq_titles = ['Seq3_u10']
    n_iter = 1000
    Seq = jnp.ones(n_iter) * 1e-3
    target_residual_schedule = jnp.full(n_iter, 10)
    n_samples_arr = [1000, 10000, 10000]
    print("MF LSVI")
    for n_samples in n_samples_arr:
        print(n_samples)
        experiment_mf_lsvi(OP_key, n_samples, n_iter, Seq, target_residual_schedule)

    """
    Doing the same but using blackjax.meanfield_vi.
    """
    sgd = 1e-3
    print("MF BLACKJAX")
    for n_samples in n_samples_arr:
        experiment(OP_key, n_iter, n_samples, sgd)
