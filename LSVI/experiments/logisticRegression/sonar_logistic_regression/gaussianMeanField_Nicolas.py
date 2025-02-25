import os
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericMeanFieldNormalDistribution, MeanFieldNormalDistribution
from variational.meanfield_gaussian_lsvi import mean_field_gaussian_lsvi

OUTPUT_PATH = "./output_mean_field"
OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def experiment(n_samples=100000, n_iter=100, lr_schedule=None, title_seq="Seq", OP_key=OP_key, OUTPUT_PATH="./output"):
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

    # Laplace Approximation for the initialisation
    # _, laplace_mean, laplace_cov = laplace_approximation(tgt_log_density, jnp.zeros(dim))
    # laplace_cov = 1 / jnp.diag(jnp.linalg.inv(laplace_cov))
    # upsilon_init = my_variational_family.get_upsilon(laplace_mean, laplace_cov)
    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim))

    if lr_schedule is None:
        lr_schedule = 1 / jnp.arange(1, n_iter + 1)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule}
    desc = "PIMA dataset, standard initialization, mean field Gaussian Nicolas"
    #if not os.path.exists(
    #        f"{OUTPUT_PATH}/gaussianMeanField_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl.pkl"):
    res, res_all = mean_field_gaussian_lsvi(OP_key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                            lr_schedule=lr_schedule)

    with open(
            f"{OUTPUT_PATH}/gaussianMeanField_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)


if __name__ == "__main__":
    n_iter = 100
    Seq_titles = ['Seq2']
    interval = jnp.arange(1, n_iter + 1)

    Seq = [1 / interval]
    Ns = [1e4]
    for idx, title in enumerate(Seq_titles):
        print(title)
        for n_samples in Ns:
            for key in range(1):
                print(key)
                print(n_samples)
                with jax.disable_jit(False):
                    experiment(n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq[idx], title_seq=title,
                               OP_key=jax.random.PRNGKey(key), OUTPUT_PATH=OUTPUT_PATH)
