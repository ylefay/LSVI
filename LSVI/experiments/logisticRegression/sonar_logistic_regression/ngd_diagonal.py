from variational.ngd import ngd_on_mf_gaussian_kl

import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericMeanFieldNormalDistribution, MeanFieldNormalDistribution

OUTPUT_PATH = "./output"
jax.config.update("jax_enable_x64", True)


def experiment(keys, n_iter, n_samples, lr, OUTPUT_PATH="./output_mean_field"):
    flipped_predictors = get_dataset(dataset="Sonar")
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_covariance = jnp.diag(my_prior_covariance)
    my_prior_log_density = MeanFieldNormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Mean Field Gaussian Variational Family
    my_variational_family = GenericMeanFieldNormalDistribution(dimension=dim)
    sampling = my_variational_family.sampling_method
    sufficient_statistic = my_variational_family.sufficient_statistic
    sanity = my_variational_family.sanity
    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim))

    @jax.vmap
    def f(key):
        res = ngd_on_mf_gaussian_kl(key, tgt_log_density, upsilon_init, n_iter, n_samples,
                        lr_schedule=lr, sanity=sanity)
        return res

    res = f(keys)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr}
    desc = "SONAR dataset, mean-field Gaussian, NGD"
    with open(
            f"{OUTPUT_PATH}/gaussian_meanfield_ngd_{n_iter}_{n_samples}_{lr if isinstance(lr, float) else "Seq"}.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': None}, f)


if __name__ == "__main__":
    n_iter = 1000
    n_samples = int(1e4)
    lr = 1 / jnp.arange(1, n_iter + 1)
    OP_key = jax.random.PRNGKey(0)
    n_repetitions = 10
    keys = jax.random.split(OP_key, n_repetitions)
    experiment(keys, n_iter, n_samples, lr, "./output_mean_field")
