import os.path
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericNormalDistribution, NormalDistribution
from variational.gaussian_lsvi import gaussian_lsvi

OUTPUT_PATH = "./output"
OP_key = jax.random.PRNGKey(4)
jax.config.update("jax_enable_x64", True)


def experiment(n_samples=100000, n_iter=100, lr_schedule=None, title_seq="Seq", OP_key=OP_key, OUTPUT_PATH="./output"):
    flipped_predictors = get_dataset(dataset="Sonar")
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Gaussian Variational Family
    my_variational_family = GenericNormalDistribution(dimension=dim)

    # Laplace Approximation for the initialisation
    #_, laplace_mean, laplace_cov = laplace_approximation(tgt_log_density, jnp.zeros(dim))
    #upsilon_init = my_variational_family.get_upsilon(laplace_mean, laplace_cov)
    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.identity(dim))

    if lr_schedule is None:
        lr_schedule = 1 / jnp.arange(1, n_iter + 1)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule}
    desc = "PIMA dataset, standard initialization, full cov. Gaussian, Nicolas"
    if not os.path.exists(
            f"{OUTPUT_PATH}/gaussian_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl"):
        res, res_all = gaussian_lsvi(OP_key, tgt_log_density, upsilon_init, n_iter, n_samples, lr_schedule=lr_schedule,
                                     return_all=False)
        with open(
                f"{OUTPUT_PATH}/gaussian_Nicolas_{n_iter}_{n_samples}_{title_seq}_{OP_key}.pkl",
                "wb") as f:
            pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)


if __name__ == "__main__":
    n_iter = int(1e2)
    Seq_titles = ['Seq2']
    interval = jnp.arange(1, n_iter + 1)

    Seq = [1 / interval]
    Ns = [1e5]
    for idx, title in enumerate(Seq_titles):
        print(title)
        for n_samples in Ns:
            for key in range(10):
                print(key)
                print(n_samples)
                experiment(n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq[idx], title_seq=title,
                           OP_key=jax.random.PRNGKey(key), OUTPUT_PATH=OUTPUT_PATH)
