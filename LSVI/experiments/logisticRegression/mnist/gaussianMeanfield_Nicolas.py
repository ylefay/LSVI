import os
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.mnist.load_mnist import mnist_dataset
from experiments.logisticRegression.utils import get_tgt_log_density
from variational.exponential_family import GenericMeanFieldNormalDistribution, MeanFieldNormalDistribution
from variational.meanfield_gaussian_lsvi import mean_field_gaussian_lsvi

OUTPUT_PATH = "./output_mean_field"
OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def experiment(keys, n_samples=100000, n_iter=100, lr_schedule=None, title_seq="Seq", OUTPUT_PATH="./output"):
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

    # Laplace Approximation for the initialisation
    # _, laplace_mean, laplace_cov = laplace_approximation(tgt_log_density, jnp.zeros(dim))
    # laplace_cov = 1 / jnp.diag(jnp.linalg.inv(laplace_cov))
    # upsilon_init = my_variational_family.get_upsilon(laplace_mean, laplace_cov)
    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim) * jnp.exp(-2))

    if lr_schedule is None:
        lr_schedule = 1 / jnp.arange(1, n_iter + 1)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule}
    desc = "MNIST dataset, mean field Gaussian Nicolas"

    @jax.vmap
    def f(key):
        res, res_all = mean_field_gaussian_lsvi(key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                                lr_schedule=lr_schedule)
        return res, res_all

    if not os.path.exists(
            f"{OUTPUT_PATH}/gaussianMeanField_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl.pkl"):
        res, res_all, = f(keys)
        with open(
                f"{OUTPUT_PATH}/gaussianMeanField_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl.pkl",
                "wb") as f:
            pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)


if __name__ == "__main__":
    n_iter = 500
    Seq_titles = ['Seq2', 'Seq3', 'Seq4']
    interval = jnp.arange(1, n_iter + 1)

    Seq = [1 / interval, jnp.maximum(1 / interval, 0.025), jnp.ones(n_iter) * 1e-3]
    Ns = [1e4]
    n_repetitions = 1
    for idx, title in enumerate(Seq_titles):
        print(title)
        for n_samples in Ns:
            for key in range(1):
                print(key)
                keys = jax.random.split(jax.random.PRNGKey(key), n_repetitions)
                print(n_samples)
                experiment(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq[idx], title_seq=title,
                           OUTPUT_PATH=OUTPUT_PATH)
