import logging

import jax.numpy as jnp
import jax.random
from timeit_decorator import timeit

from experiments.syntheticLikelihood.heuristic_gaussian import experiment as heuristic_gaussian
from experiments.syntheticLikelihood.heuristic_truncated_gaussian import experiment as heuristic_truncated_gaussian
from experiments.syntheticLikelihood.rwmh import experiment as rwmh

logging.basicConfig(level=logging.INFO)
jax.config.update("jax_enable_x64", True)
OUTPUT_PATH = "./output_timeit"


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussian():
    OP_key = jax.random.PRNGKey(0)
    n_iter = 50
    Seq_title = 'inv_u_1_shrinkage_05'
    interval = jnp.arange(1, n_iter + 1)
    Seq = 1 / interval
    n_samples = 100
    target_residual_schedule = 1.0
    heuristic_gaussian(n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq, title_seq=Seq_title,
                       target_residual_schedule=target_residual_schedule,
                       OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_heuristic_truncated_gaussian():
    OP_key = jax.random.PRNGKey(0)
    n_iter = 100
    Seq_title = 'inv_u_1_shrinkage_05'
    interval = jnp.arange(1, n_iter + 1)
    target_residual_schedule = 1.0
    Seq = 1 / interval
    n_samples = 100
    heuristic_truncated_gaussian(n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq, title_seq=Seq_title,
                                 target_residual_schedule=target_residual_schedule,
                                 OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_rwmh():
    OP_key = jax.random.PRNGKey(0)
    n_samples_for_tgt_log = 100
    num_mcmc_steps = 10000
    rwmh(OP_key, num_mcmc_steps, n_samples_for_tgt_log, OUTPUT_PATH=OUTPUT_PATH)


if __name__ == "__main__":
    time_heuristic_gaussian()
    time_heuristic_truncated_gaussian()
    time_rwmh()
