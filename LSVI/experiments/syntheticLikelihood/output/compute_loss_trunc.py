import os
import pickle

import jax
import jax.numpy as jnp

from experiments.syntheticLikelihood.fowler_toad import get_tgt_density, simulation
from variational.exponential_family import GenericTruncatedMFNormalDistribution
from variational.utils import truncated_mf_gaussian_loss

OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)
OUTPUT = "./losses"
EXCLUDED_PICKLES = ["sdata_[0 0].pkl", "sdata_[0 1].pkl", "sdata_[0 2].pkl"]
SIZE_vmap = 11
OUTPUT_PATH = "./"

"""
Compute the opposite of the ELBO for the Gaussian variational family
"""

if __name__ == "__main__":
    dim = 3
    truncated_gaussian = GenericTruncatedMFNormalDistribution(dimension=dim, lower=jnp.array([1., 0., 0.]),
                                                              upper=jnp.array([2., 100., 0.9]))
    n_sample_ll = 10
    num_days = 17
    num_tods = 66
    PKLs = []
    PKL_titles = []
    for file in os.listdir("./"):
        if file.endswith(".pkl"):
            PKLs.append(pickle.load(open(file, "rb")))
            PKL_titles.append(str(file))

    for idx, my_pkl in enumerate(PKLs):
        for key_idx in range(3):
            if PKL_titles[idx] not in EXCLUDED_PICKLES and PKL_titles[idx][
                                                           -7:] == f" {key_idx}].pkl" and "ht_data" not in PKL_titles[
                idx]:
                key = jax.random.PRNGKey(key_idx)
                size_pkl = my_pkl['res'].shape[0]
                pickle.load(open(f"{OUTPUT_PATH}/ht_data_{size_pkl - 1}_1000_Seq2_u_166_01_A_{key}.pkl", "rb"))
                sdata = simulation(OP_key, num_days, num_tods, jnp.array([1.7, 35, 0.6]))  # 63, 66
                _tgt_log_density = get_tgt_density(sdata, n_sample_ll)
                tgt_log_density = lambda x: _tgt_log_density(OP_key, x)


                @jax.vmap
                def wrapper_truncated_gaussian_loss(key, theta):
                    # Wrapper for the loss -ELBO(q_\upsilon \mid \pi) where q_\upsilon is a full-rank Gaussian distribution
                    return truncated_mf_gaussian_loss(OP_key=key, theta=theta, truncated_gaussian=truncated_gaussian,
                                                      tgt_log_density=tgt_log_density, n_samples_for_loss=int(1e2))


                loss = jnp.zeros(size_pkl)
                keys = jax.random.split(OP_key, size_pkl // SIZE_vmap + 1)
                for k in range(size_pkl // SIZE_vmap):
                    keys2 = jax.random.split(keys[k], SIZE_vmap)
                    loss = loss.at[k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl)].set(
                        wrapper_truncated_gaussian_loss(keys2,
                                                        my_pkl['res'][k * SIZE_vmap:min((k + 1) * SIZE_vmap, size_pkl),
                                                        :-1]))
                if size_pkl % SIZE_vmap != 0:
                    keys2 = jax.random.split(keys[-1], size_pkl % SIZE_vmap)
                    loss = loss.at[-(size_pkl % SIZE_vmap):].set(
                        wrapper_truncated_gaussian_loss(keys2, my_pkl['res'][-(size_pkl % SIZE_vmap):, :-1]))
                with open(f"{OUTPUT}/{PKL_titles[idx][:-4]}_loss.pkl", "wb") as f:
                    pickle.dump(loss, f)
