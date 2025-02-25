import pickle

import jax
import jax.numpy as jnp

from experiments.syntheticLikelihood.fowler_toad import simulation

OUTPUT_PATH = "./output/"
OP_key = jax.random.PRNGKey(0)
num_days = 63
num_tods = 66
params = jnp.array([1.7, 35, 0.6])

if __name__ == "__main__":
    sdata = simulation(OP_key, num_days, num_tods, params)
    pickle.dump(sdata, open(f"{OUTPUT_PATH}/ht_data_{num_days}_{num_tods}_{OP_key}.pkl", "wb"))
