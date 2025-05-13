import pickle

import jax
import jax.numpy as jnp
import numpy as np

from experiments.logisticRegression.subsampling_review.get_dataset import get_Census_Income_dataset
from experiments.logisticRegression.subsampling_review.utils import get_tgt_log_density

OUTPUT_PATH = "./output_mean_field"
from experiments.time_wrapper import timer
from variational.exponential_family import GenericMeanFieldNormalDistribution, NormalDistribution
from variational.exponential_family import MeanFieldNormalDistribution
from variational.meanfield_gaussian_lsvi import mean_field_gaussian_lsvi
from experiments.logisticRegression.subsampling_review.ngd import ngd_on_mf_gaussian_kl
from experiments.logisticRegression.subsampling_review.heuristic_and_non_heuristic_gaussianMeanField_NIcolas import \
    mean_field_gaussian_lsvi

n_runs = 5

OP_key = jax.random.PRNGKey(0)

jax.config.update("jax_enable_x64", True)


@timer(runs=n_runs)
def experiment(key, n_samples=100000, n_iter=100, lr_schedule=None, target_residual_schedule=None):
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
    desc = "Census dataset, heuristic, mf. Gaussian, Nicolas"

    # if not os.path.exists(
    #        f"{OUTPUT_PATH}/heuristic_gaussian_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl"):

    def f(key):
        res, res_all = mean_field_gaussian_lsvi(key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                                lr_schedule=lr_schedule,
                                                target_residual_schedule=target_residual_schedule,
                                                return_all=False)
        return res, res_all

    res, res_all = f(key)
    return None


@timer(runs=n_runs)
def experiment_ngd(key, n_iter, n_samples, lr_schedule):
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

    def f(key):
        res = ngd_on_mf_gaussian_kl(key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                    lr_schedule=lr_schedule, sanity=sanity)
        return res

    res = f(key)
    return None


if __name__ == "__main__":
    """
    Running n_runs time with timeit decorator the experiment mean field lsvi
    """
    n_iter = 100
    Seq = jnp.ones(n_iter) * 1e-3
    target_residual_schedule = jnp.full(n_iter, 10)
    n_samples_arr = [10000]

    time_results = np.zeros((3, len(n_samples_arr), n_runs))

    print("MF LSVI (sch 6, 10 1e-3)")
    for idx, n_samples in enumerate(n_samples_arr):
        print(n_samples)
        time_results[0, idx] = experiment(OP_key, n_samples, n_iter, Seq, target_residual_schedule)

    target_residual_schedule = jnp.inf
    print("MF LSVI (sch 5, inf 1e-3)")
    for idx, n_samples in enumerate(n_samples_arr):
        print(n_samples)
        time_results[1, idx] = experiment(OP_key, n_samples, n_iter, Seq, target_residual_schedule)

    """
    NGD
    sch. 5 (u=inf, eps=1e-3)
    """
    for idx, n_samples in enumerate(n_samples_arr):
        print(n_samples)
        time_results[4, idx] = experiment_ngd(OP_key, n_iter, n_samples, Seq)

    with open("walltime_results.pkl", "wb") as f:
        pickle.dump(time_results, f)
