import pickle

import jax
import jax.numpy as jnp
import optax
from blackjax.vi import meanfield_vi

from experiments.logisticRegression.mnist.load_mnist import mnist_dataset
from experiments.logisticRegression.utils import get_tgt_log_density
from variational.exponential_family import MeanFieldNormalDistribution

OUTPUT_PATH = "./output_answer_to_secondround"
OP_key = jax.random.PRNGKey(0)

jax.config.update("jax_enable_x64", True)


def experiment(keys, num_iter, num_samples, sgd=1e-3, OUTPUT_PATH="./output_mean_field", s=""):
    flipped_predictors = mnist_dataset(return_test=False)
    dim = flipped_predictors.shape[1]

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    # my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_covariance = jnp.diag(my_prior_covariance)
    my_prior_log_density = MeanFieldNormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)
    opt = optax.sgd(sgd)
    res = meanfield_vi.as_top_level_api(tgt_log_density, optimizer=opt, num_samples=num_samples)
    # initial_state = {"mu": jnp.zeros(dim), "rho": jnp.zeros(dim)}
    initial_state = res.init(position=jnp.zeros(dim))

    @jax.vmap
    def inference_loop(rng_key):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = res.step(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_iter)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        mus = jnp.array(states.mu)
        mus = jnp.insert(mus, 0, jnp.array(initial_state.mu), 0)
        rhos = jnp.array(states.rho)
        rhos = jnp.insert(rhos, 0, jnp.array(initial_state.rho), 0)

        return mus, rhos

    states = inference_loop(keys)
    with open(f"{OUTPUT_PATH}/res_mfg_advi_blackjax_{s}_{num_iter}_{num_samples}_{sgd}.pkl", "wb") as f:
        pickle.dump(
            {'desc': "MNIST dataset, mean field Gaussian ADVI blackjax", 'num_iter': num_iter,
             'num_samples': num_samples, 'sgd': sgd, 'states': (states[0], states[1])}, f)


if __name__ == "__main__":
    N_iters = [100]
    sgd = 1e-3
    num_samples = int(1e4)
    n_repetitions = 10
    for num_iter in N_iters:
        print(num_iter)
        for key in range(10):
            keys = jax.random.split(jax.random.PRNGKey(key), n_repetitions)
            s = key
            experiment(keys, int(num_iter), num_samples, sgd, s)
