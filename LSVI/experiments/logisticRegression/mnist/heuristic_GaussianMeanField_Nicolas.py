import os.path
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.mnist.load_mnist import mnist_dataset
from experiments.logisticRegression.utils import get_tgt_log_density
from variational.exponential_family import GenericMeanFieldNormalDistribution, NormalDistribution
from variational.meanfield_gaussian_lsvi import mean_field_gaussian_lsvi

OUTPUT_PATH = "./output_mean_field"
OP_key = jax.random.PRNGKey(4)
jax.config.update("jax_enable_x64", True)


def experiment(n_samples=100000, n_iter=100, lr_schedule=None, target_residual_schedule=None, title_seq="Seq",
               OP_key=OP_key, OUTPUT_PATH="./output"):
    flipped_predictors = mnist_dataset(return_test=False)
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Mean Field Gaussian Variational Family
    my_variational_family = GenericMeanFieldNormalDistribution(dimension=dim)

    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim) * jnp.exp(-2))

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule, 'residual': target_residual_schedule}
    desc = "MNIST dataset, heuristic, mf. Gaussian, Nicolas"
    if not os.path.exists(
            f"{OUTPUT_PATH}/heuristic_gaussian_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl"):
        with jax.disable_jit(False):
            res, res_all = mean_field_gaussian_lsvi(OP_key, tgt_log_density, upsilon_init, n_iter, n_samples,
                                                    lr_schedule=lr_schedule,
                                                    target_residual_schedule=target_residual_schedule,
                                                    return_all=False)
        with open(
                f"{OUTPUT_PATH}/heuristic_gaussian_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl",
                "wb") as f:
            pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)


if __name__ == "__main__":
    n_iter = 5
    Seq_titles = ['Seq1_u10', 'Seq3_u10']
    interval = jnp.arange(1, n_iter + 1)
    Seq = [jnp.ones(n_iter), jnp.ones(n_iter) * 1e-3]
    Ns = [1e4]
    for idx, title in enumerate(Seq_titles):
        print(title)
        for n_samples in Ns:
            target_residual_schedule = jnp.full(n_iter, 10)
            for key in range(1):
                print(key)
                print(n_samples)
                experiment(n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq[idx],
                           target_residual_schedule=target_residual_schedule, title_seq=title,
                           OP_key=jax.random.PRNGKey(key), OUTPUT_PATH=OUTPUT_PATH)
