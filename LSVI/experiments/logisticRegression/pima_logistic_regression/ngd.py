import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericNormalDistribution, NormalDistribution
from variational.ngd import ngd_on_gaussian_kl

OUTPUT_PATH = "./ngd"
jax.config.update("jax_enable_x64", True)


def experiment(keys, n_iter, n_samples, lr, OUTPUT_PATH="./output"):
    flipped_predictors = get_dataset()
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

    @jax.vmap
    def f(key):
        return ngd_on_gaussian_kl(key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                  lr_schedule=lr, sanity=sanity)

    res = f(keys)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr}
    desc = "PIMA dataset, full cov. Gaussian, NGD"
    with open(
            f"{OUTPUT_PATH}/gaussian_ngd_{n_iter}_{n_samples}_{lr if isinstance(lr, float) else "Seq"}_inv.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': None}, f)


if __name__ == "__main__":
    n_iter = 100
    n_samples = int(1e4)
    lr = 1 / jnp.arange(1, n_iter + 1)
    OP_key = jax.random.PRNGKey(1)
    number_of_repetition = 100
    keys = jax.random.split(OP_key, number_of_repetition)
    experiment(keys, n_iter, n_samples, lr, "./ngd")
