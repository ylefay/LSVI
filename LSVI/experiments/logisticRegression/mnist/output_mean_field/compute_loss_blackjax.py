import os
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.mnist.load_mnist import mnist_dataset
from experiments.logisticRegression.utils import get_tgt_log_density
from variational.exponential_family import MeanFieldNormalDistribution, GenericMeanFieldNormalDistribution
from variational.utils import gaussian_loss

OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)
OUTPUT = "./losses"
EXCLUDED_PICKLES = []
SIZE_vmap = 5

"""
Compute the opposite of the ELBO for the Gaussian variational family
"""

if __name__ == "__main__":

    # Compute the log density
    flipped_predictors = mnist_dataset(path_prefix="..")
    N, dim = flipped_predictors.shape
    my_prior_covariance = 25 * jnp.identity(dim)
    # my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_covariance = jnp.diag(my_prior_covariance)
    my_prior_log_density = MeanFieldNormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)
    mfg_gaussian = GenericMeanFieldNormalDistribution(dimension=dim)
    # Read pickles
    PKLs = []
    PKL_titles = []
    for file in os.listdir("./"):
        if file.endswith(".pkl"):
            PKLs.append(pickle.load(open(file, "rb")))
            PKL_titles.append(str(file))


    @jax.vmap
    def wrapper_gaussian_loss(key, res):
        # Wrapper for the loss -ELBO(q_\upsilon \mid \pi) where q_\upsilon is a full-rank Gaussian distribution
        locs = res[0]
        sigmassq = jnp.exp(res[1])
        theta = mfg_gaussian.get_theta(locs, sigmassq)
        return gaussian_loss(OP_key=key, theta=theta, gaussian=mfg_gaussian,
                             tgt_log_density=tgt_log_density, n_samples_for_loss=int(1e4))


    for idx, my_pkl in enumerate(PKLs):
        if PKL_titles[idx] not in EXCLUDED_PICKLES and "blackjax" in PKL_titles[idx]:
            if not os.path.exists(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl"):
                n_repeat = my_pkl['states'][0].shape[0]
                size_pkl = my_pkl['states'][0].shape[1]
                loss = jnp.zeros((n_repeat, size_pkl))
                keys = jax.random.split(OP_key, (size_pkl // SIZE_vmap + 1) * n_repeat).reshape(
                    (n_repeat, size_pkl // SIZE_vmap + 1, -1))
                for repeat in range(n_repeat):
                    for k in range(size_pkl // SIZE_vmap):
                        keys2 = jax.random.split(keys[repeat, k], SIZE_vmap)
                        c1 = my_pkl['states'][0][repeat, k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl)]
                        c2 = my_pkl['states'][1][repeat, k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl)]
                        loss = loss.at[repeat, k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl)].set(
                            wrapper_gaussian_loss(keys2, (c1, c2)))
                    if size_pkl % SIZE_vmap != 0:
                        keys2 = jax.random.split(keys[repeat, -1], size_pkl % SIZE_vmap)
                        c1 = my_pkl['states'][0][repeat, -(size_pkl % SIZE_vmap):]
                        c2 = my_pkl['states'][1][repeat, -(size_pkl % SIZE_vmap):]
                        loss = loss.at[repeat, -(size_pkl % SIZE_vmap):].set(
                            wrapper_gaussian_loss(keys2, (c1, c2)))
                with open(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl", "wb") as f:
                    pickle.dump(loss, f)
