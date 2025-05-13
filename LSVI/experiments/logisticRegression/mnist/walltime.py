import pickle

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from blackjax.vi import meanfield_vi

from experiments.logisticRegression.mnist.load_mnist import mnist_dataset
from experiments.logisticRegression.utils import get_tgt_log_density
from experiments.time_wrapper import timer
from variational.exponential_family import GenericMeanFieldNormalDistribution, NormalDistribution
from variational.exponential_family import MeanFieldNormalDistribution
from variational.meanfield_gaussian_lsvi import mean_field_gaussian_lsvi
from variational.ngd import ngd

n_runs = 5

OP_key = jax.random.PRNGKey(0)

jax.config.update("jax_enable_x64", True)


@timer(runs=n_runs)
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


@timer(runs=n_runs)
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


@timer(runs=n_runs)
def experiment_ngd(key, n_iter, n_samples, lr):
    flipped_predictors = mnist_dataset(return_test=False)
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    # my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_covariance = jnp.diag(my_prior_covariance)
    my_prior_log_density = MeanFieldNormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Mean Field Gaussian Variational Family
    my_variational_family = GenericMeanFieldNormalDistribution(dimension=dim)
    sampling = my_variational_family.sampling_method
    sufficient_statistic = my_variational_family.sufficient_statistic
    sanity = my_variational_family.sanity

    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim) * jnp.exp(-2))

    def f(key):
        res = ngd(key, sampling, sufficient_statistic, tgt_log_density, upsilon_init, n_iter, n_samples,
                  lr_schedule=lr, sanity=sanity)
        return res

    res = f(key)
    return None


if __name__ == "__main__":
    """
    Running n_runs time with timeit decorator the experiment mean field lsvi for different n_samples
    """
    n_iter = 100
    Seq = jnp.ones(n_iter) * 1e-3
    target_residual_schedule = jnp.full(n_iter, 10)
    n_samples_arr = [1000, 10000, 50000, 100000]

    time_results = np.zeros((5, len(n_samples_arr), n_runs))

    print("MF LSVI (sch 6, 10 1e-3)")
    for idx, n_samples in enumerate(n_samples_arr):
        print(n_samples)
        time_results[0, idx] = experiment_mf_lsvi(OP_key, n_samples, n_iter, Seq, target_residual_schedule)

    target_residual_schedule = jnp.inf
    print("MF LSVI (sch 5, inf 1e-3)")
    for idx, n_samples in enumerate(n_samples_arr):
        print(n_samples)
        time_results[1, idx] = experiment_mf_lsvi(OP_key, n_samples, n_iter, Seq, target_residual_schedule)

    """
    Doing the same but using blackjax.meanfield_vi.
    """
    sgd = 1e-3
    print("MF BLACKJAX")
    for idx, n_samples in enumerate(n_samples_arr):
        print(n_samples)
        time_results[2, idx] = experiment(OP_key, n_iter, n_samples, sgd)
    Seq = 1 / jnp.arange(1, n_iter + 1) * 1e-3
    n_samples_arr = [10000]
    """
    Forgot sch. 4 (1, 1)
    """
    print("MF LSVI (sch 4)")
    Seq = jnp.ones(n_iter)
    target_residual_schedule = jnp.full(n_iter, 1)
    for idx, n_samples in enumerate(n_samples_arr):
        print(n_samples)
        time_results[3, idx] = experiment_mf_lsvi(OP_key, n_samples, n_iter, Seq, target_residual_schedule)

    """
    NGD
    sch. 5 (u=inf, eps=1e-3)
    """
    for idx, n_samples in enumerate(n_samples_arr):
        print(n_samples)
        time_results[4, idx] = experiment_ngd(OP_key, n_iter, n_samples, Seq)

    with open("walltime_results.pkl", "wb") as f:
        pickle.dump(time_results, f)
