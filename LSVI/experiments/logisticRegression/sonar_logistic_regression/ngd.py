from variational.ngd import ngd_on_gaussian_kl

import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericNormalDistribution, NormalDistribution

OUTPUT_PATH = "./output"
jax.config.update("jax_enable_x64", True)


def experiment(OP_key, n_iter, n_samples, lr, OUTPUT_PATH="./output"):
    flipped_predictors = get_dataset(dataset="Sonar")
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Gaussian Variational Family
    my_variational_family = GenericNormalDistribution(dimension=dim)
    sampling = my_variational_family.sampling_method
    sufficient_statistic = my_variational_family.sufficient_statistic
    sanity = my_variational_family.sanity

    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.identity(dim))

    res = ngd_on_gaussian_kl(OP_key, tgt_log_density, upsilon_init, n_iter, n_samples,
                        lr_schedule=lr, sanity=sanity)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr}
    desc = "SONAR dataset, full-cov Gaussian, NGD"
    with open(
            f"{OUTPUT_PATH}/gaussian_ngd_{n_iter}_{n_samples}_{lr if isinstance(lr, float) else "Seq"}.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': None}, f)


if __name__ == "__main__":
    n_iter = 100
    n_samples = int(1e4)
    lr = 1 / jnp.arange(1, n_iter + 1)
    OP_key = jax.random.PRNGKey(0)
    experiment(OP_key, n_iter, n_samples, lr, "./output")
