import pickle
from typing import Callable

import blackjax.mcmc.random_walk as rw
import jax
import jax.numpy as jnp
from blackjax import additive_step_random_walk
from jax.typing import ArrayLike

from experiments.syntheticLikelihood.fowler_toad import get_tgt_density


def repeat(a: ArrayLike, n: int) -> ArrayLike:
    return jnp.repeat(a[jnp.newaxis, ...], n, axis=0)


class RWMHAdaptative:
    """
    Random Walk Metropolis-Hastings with adaptative covariance matrix learn from the previous iterates.
    """

    def __init__(self, log_tgt_density: Callable[[jax.Array, jnp.ndarray], jnp.ndarray], dim: int):
        """
        log_tgt_density: log target density function, must be a function of the form f: key, x -> jnp.ndarray
        In case your log-density is not random, please provide f = lambda key, x: log_tgt_density(x)
        """
        self.log_tgt_density = log_tgt_density
        self.dim = dim

    def run(self, key: jax.Array, initial_position: jnp.ndarray, num_mcmc_steps: int, burnin_period: int,
            sigma: jnp.ndarray):
        """
        initial_position: initial position of the chain
        num_mcmc_steps: number of steps of the chain
        burnin_period: number of steps before starting to adapt the covariance matrix
        sigma: initial covariance matrix, used before the burnin period
        Returns
        -------
        chain: the chain of states Tuple[RWState, RWInfo]
        """
        log_tgt_density = lambda x: self.log_tgt_density(key, x)
        init_state = rw.init(position=initial_position, logdensity_fn=log_tgt_density)

        def get_cov(cumul_positions):
            return 2.38 / self.dim * jnp.cov(cumul_positions, rowvar=False)

        def body_fn(carry, key):
            i, state, cumul_states = carry
            log_tgt_density = lambda x: self.log_tgt_density(key, x)
            cov = jax.lax.cond(i > burnin_period, lambda _: get_cov(cumul_states), lambda _: sigma, None)
            random_walk = additive_step_random_walk(log_tgt_density, rw.normal(cov))
            state = random_walk.init(state.position)
            new_state, info = random_walk.step(key, state)
            cumul_states = cumul_states.at[i].set(new_state.position)

            return (i + 1, new_state, cumul_states), (new_state, info)

        (_, _, _), chain = jax.lax.scan(body_fn, (0, init_state, repeat(init_state.position, num_mcmc_steps)),
                                        jax.random.split(key, num_mcmc_steps))

        return chain


class RWMHFixedSigma:
    """
    Similar to RWMH_adaptative with fixed covariance matrix
    """

    def __init__(self, log_tgt_density: Callable, dim: int):
        self.log_tgt_density = log_tgt_density
        self.dim = dim

    def run(self, key: jax.Array, initial_positions: jnp.ndarray, num_mcmc_steps: int, sigma: jnp.ndarray):
        log_tgt_density = lambda x: self.log_tgt_density(key, x)
        init_states = rw.init(position=initial_positions, logdensity_fn=log_tgt_density)

        def body_fn(carry, key):
            i, state = carry
            log_tgt_density = lambda x: self.log_tgt_density(key, x)
            random_walk = additive_step_random_walk(log_tgt_density, rw.normal(sigma))
            state = random_walk.init(state.position)
            new_state, info = random_walk.step(key, state)
            return (i + 1, new_state), (new_state, info)

        (_, _), chain = jax.lax.scan(body_fn, (0, init_states), jax.random.split(key, num_mcmc_steps))

        return chain


def experiment(OP_key, num_mcmc_steps, n_samples_for_tgt_log, OUTPUT_PATH="./output"):
    dim = 3
    sigma = jnp.eye(3) * 0.1
    initial_position = jnp.zeros(3)
    sdata = pickle.load(open(f"{OUTPUT_PATH}/ht_data_63_66_[0 0].pkl", "rb"))

    _tgt_log_density = get_tgt_density(sdata, n_samples_for_tgt_log, shrinkage=0.5, transform=True, scales2=jnp.ones(3))
    tgt_log_density = lambda key, x: _tgt_log_density(key, jnp.expand_dims(x, 0)).at[0].get()

    my_rwmh = RWMHFixedSigma(log_tgt_density=tgt_log_density, dim=dim)
    chain = my_rwmh.run(OP_key, initial_position, num_mcmc_steps, sigma)
    with open(f"{OUTPUT_PATH}/rwmh_constant_sigma_{n_samples_for_tgt_log}_{num_mcmc_steps}_{OP_key}.pkl",
              "wb") as f:
        pickle.dump({'chain': chain}, f)


if __name__ == "__main__":
    OUTPUT_PATH = "./output"
    OP_key = jax.random.PRNGKey(4)
    jax.config.update("jax_enable_x64", True)
    n_samples_for_tgt_log = 100
    num_mcmc_steps = 10000
    experiment(OP_key, num_mcmc_steps, n_samples_for_tgt_log, OUTPUT_PATH=OUTPUT_PATH)
