import os
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_tgt_log_density, get_dataset
from variational.exponential_family import NormalDistribution, GenericNormalDistribution
from variational.utils import gaussian_loss

OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)
OUTPUT = "./losses"
EXCLUDED_PICKLES = ["res_LP_plus_Gaussian_100_10000_Seq.pkl",
                    "res_LP_plus_Gaussian_Nicolas_100_100000_Seq1_[0 2].pkl","res_LP_plus_Gaussian_Nicolas_100_100000_Seq3.pkl"]
SIZE_vmap = 11
#"res_LP_plus_Gaussian_100_10000_1.0.pkl"
"""
Compute the opposite of the ELBO for the Gaussian variational family
"""

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
    def wrapper_gaussian_loss(key, theta):
        # Wrapper for the loss -ELBO(q_\upsilon \mid \pi) where q_\upsilon is a full-rank Gaussian distribution
        return gaussian_loss(OP_key=key, theta=theta, gaussian=full_gaussian,
                             tgt_log_density=tgt_log_density, n_samples_for_loss=int(1e4))


    for idx, my_pkl in enumerate(PKLs):
        if PKL_titles[idx] not in EXCLUDED_PICKLES:
            size_pkl = my_pkl['res'].shape[0]
            loss = jnp.zeros(size_pkl)
            keys = jax.random.split(OP_key, size_pkl // SIZE_vmap + 1)
            for k in range(size_pkl // SIZE_vmap):
                keys2 = jax.random.split(keys[k], SIZE_vmap)
                loss = loss.at[k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl)].set(
                    wrapper_gaussian_loss(keys2, my_pkl['res'][k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl), :-1]))
            if size_pkl % SIZE_vmap != 0:
                keys2 = jax.random.split(keys[-1], size_pkl % SIZE_vmap)
                loss = loss.at[-(size_pkl % SIZE_vmap):].set(
                    wrapper_gaussian_loss(keys2, my_pkl['res'][-(size_pkl % SIZE_vmap):, :-1]))
            with open(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl", "wb") as f:
                pickle.dump(loss, f)
