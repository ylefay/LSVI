import os
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.mnist.load_mnist import mnist_dataset_full
from experiments.logisticRegression.utils import multilogistic_get_tgt_log_density as get_tgt_log_density
from variational.exponential_family import GenericMeanFieldNormalDistribution, MeanFieldNormalDistribution
from variational.utils import gaussian_loss

OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)
OUTPUT = "./losses"
EXCLUDED_PICKLES = []
SIZE_vmap = 40

"""
Compute the opposite of the ELBO for the Gaussian variational family
"""

if __name__ == "__main__":

    # Compute the log density
    predictors, labels = mnist_dataset_full(return_test=False, path_prefix="../..")
    K = labels.shape[1]
    N, dim = predictors.shape
    latent_dim = dim * K
    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(latent_dim)
    # my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_covariance = jnp.diag(my_prior_covariance)
    my_prior_log_density = MeanFieldNormalDistribution(jnp.zeros(latent_dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(predictors, labels, my_prior_log_density)

    # Mean Field Gaussian Variational Family
    mfg_gaussian = GenericMeanFieldNormalDistribution(dimension=latent_dim)
    # Read pickles
    PKLs = []
    PKL_titles = []
    for file in os.listdir("./"):
        if file.endswith(".pkl"):
            PKLs.append(pickle.load(open(file, "rb")))
            PKL_titles.append(str(file))


    @jax.vmap
    def wrapper_gaussian_loss(key, theta):
        # Wrapper for the loss -ELBO(q_\upsilon \mid \pi) where q_\upsilon is a full-rank Gaussian distribution
        return gaussian_loss(OP_key=key, theta=theta, gaussian=mfg_gaussian,
                             tgt_log_density=tgt_log_density, n_samples_for_loss=int(1e4))


    for idx, my_pkl in enumerate(PKLs):
        if PKL_titles[idx] not in EXCLUDED_PICKLES:
            if not os.path.exists(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl"):
                size_pkl = my_pkl['res'].shape[0]
                loss = jnp.zeros(size_pkl)
                keys = jax.random.split(OP_key, size_pkl // SIZE_vmap + 1)
                for k in range(size_pkl // SIZE_vmap):
                    keys2 = jax.random.split(keys[k], SIZE_vmap)
                    loss = loss.at[k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl)].set(
                        wrapper_gaussian_loss(keys2,
                                              my_pkl['res'][k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl), :-1]))
                if size_pkl % SIZE_vmap != 0:
                    keys2 = jax.random.split(keys[-1], size_pkl % SIZE_vmap)
                    loss = loss.at[-(size_pkl % SIZE_vmap):].set(
                        wrapper_gaussian_loss(keys2, my_pkl['res'][-(size_pkl % SIZE_vmap):, :-1]))
                with open(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl", "wb") as f:
                    pickle.dump(loss, f)
