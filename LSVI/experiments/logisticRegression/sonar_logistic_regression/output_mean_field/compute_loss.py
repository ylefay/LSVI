import os
import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_tgt_log_density, get_dataset
from variational.exponential_family import MeanFieldNormalDistribution, GenericMeanFieldNormalDistribution
from variational.utils import gaussian_loss

OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)
OUTPUT = "./losses"
EXCLUDED_PICKLES = []

"""
Compute the opposite of the ELBO for the Gaussian variational family
"""

if __name__ == "__main__":

    # Compute the log density
    flipped_predictors = get_dataset(dataset="Sonar")
    N, dim = flipped_predictors.shape
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
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
    def wrapper_gaussian_loss(key, theta):
        # Wrapper for the loss -ELBO(q_\upsilon \mid \pi) where q_\upsilon is a full-rank Gaussian distribution
        return gaussian_loss(OP_key=key, theta=theta, gaussian=mfg_gaussian,
                             tgt_log_density=tgt_log_density, n_samples_for_loss=int(1e4))


    """
    NEED TO CHECK THIS ONE.
    """

    for idx, my_pkl in enumerate(PKLs):
        if PKL_titles[idx] not in EXCLUDED_PICKLES:
            if not os.path.exists(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl"):
                size_pkl = my_pkl['res'].shape[1]
                n_repeat = my_pkl['res'].shape[0]
                loss = jnp.zeros((n_repeat, size_pkl))
                keys = jax.random.split(OP_key, n_repeat * (size_pkl)).reshape((n_repeat, size_pkl, -1))
                for repeat in range(n_repeat):
                    loss = loss.at[repeat].set(wrapper_gaussian_loss(keys[repeat], my_pkl['res'][repeat, :, :-1]))
                with open(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl", "wb") as f:
                    pickle.dump(loss, f)
