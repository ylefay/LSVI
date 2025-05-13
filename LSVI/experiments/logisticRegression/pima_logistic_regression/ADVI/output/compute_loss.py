import os
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_tgt_log_density, get_dataset
from variational.exponential_family import NormalDistribution, GenericNormalDistribution
from variational.utils import gaussian_loss, elbo_integrand

OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)
OUTPUT = "./losses"
EXCLUDED_PICKLES = [""]
SIZE_vmap = 11

"""
Compute the opposite of the ELBO for the Gaussian variational family
"""


def gaussian_loss(OP_key, meancov, gaussian, tgt_log_density, n_samples_for_loss=100):
    """
    Compute the ELBO for a Gaussian variational family.
    """
    mean = meancov.at[0].get()
    cov = meancov.at[1:].get()

    def logpdf_normal(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)

    def my_integrand(x):
        return elbo_integrand(logpdf_normal, tgt_log_density, x)

    samples = jax.random.multivariate_normal(OP_key, mean, cov, (n_samples_for_loss,))
    integral = jnp.sum(jax.vmap(my_integrand)(samples)) / n_samples_for_loss
    return integral


if __name__ == "__main__":

    # Compute the log density
    flipped_predictors = get_dataset(dataset="Pima")
    N, dim = flipped_predictors.shape
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)
    full_gaussian = GenericNormalDistribution(dimension=dim)
    # Read pickles
    PKLs = []
    PKL_titles = []
    for file in os.listdir("./"):
        if file.endswith(".pkl"):
            PKLs.append(pickle.load(open(file, "rb")))
            PKL_titles.append(str(file))


    @jax.vmap
    def wrapper_gaussian_loss(key, meancov):
        # Wrapper for the loss -ELBO(q_\upsilon \mid \pi) where q_\upsilon is a full-rank Gaussian distribution
        return gaussian_loss(OP_key=key, meancov=meancov, gaussian=full_gaussian,
                             tgt_log_density=tgt_log_density, n_samples_for_loss=int(1e4))


    for idx, my_pkl in enumerate(PKLs):
        if PKL_titles[idx] not in EXCLUDED_PICKLES:
            n_repetitions = my_pkl['means'].shape[0]
            size_pkl = my_pkl['means'].shape[1]
            loss = jnp.zeros((n_repetitions, size_pkl))
            keys = jax.random.split(OP_key, (size_pkl // SIZE_vmap + 1) * n_repetitions).reshape(
                (n_repetitions, size_pkl // SIZE_vmap + 1, -1))
            for repeat in range(n_repetitions):
                my_means = jnp.array(my_pkl['means'][repeat])[:, jnp.newaxis]
                my_covs = jnp.array(my_pkl['covs'][repeat])
                my_meanscovs = jnp.concatenate([my_means, my_covs], axis=1)
                for k in range(my_means.shape[0] // SIZE_vmap):
                    keys2 = jax.random.split(keys[repeat, k], SIZE_vmap)
                    loss = loss.at[repeat, k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl)].set(
                        wrapper_gaussian_loss(keys2, my_meanscovs[k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl)]))
                if size_pkl % SIZE_vmap != 0:
                    keys2 = jax.random.split(keys[repeat, -1], size_pkl % SIZE_vmap)
                    loss = loss.at[repeat, -(size_pkl % SIZE_vmap):].set(
                        wrapper_gaussian_loss(keys2, my_meanscovs[-(size_pkl % SIZE_vmap):]))
            with open(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl", "wb") as f:
                pickle.dump(loss, f)
