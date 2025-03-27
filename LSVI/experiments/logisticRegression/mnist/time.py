import logging

import jax.numpy as jnp
import jax.random
from timeit_decorator import timeit

from experiments.logisticRegression.mnist.gaussianMeanField_ADVI_blackjax import \
    experiment as gaussianMeanField_ADVI_blackjax
from experiments.logisticRegression.mnist.gaussianMeanfield_Nicolas import experiment as gaussianMeanField_Nicolas
from experiments.logisticRegression.mnist.heuristic_GaussianMeanField_Nicolas import \
    experiment as heuristic_gaussianMeanField_Nicolas
from experiments.logisticRegression.mnist.ngd_diagonal import experiment as ngd_diagonal

logging.basicConfig(level=logging.INFO)
jax.config.update("jax_enable_x64", True)
OUTPUT_PATH = "./output_timeit"

n_runs = 1
n_repetitions = 1
OP_key = jax.random.PRNGKey(0)
keys = jax.random.split(OP_key, n_repetitions)

@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussianMeanField_Nicolas_u10_fixed1em3():
    n_iter = int(5e2)
    Seq_title = '1em3_u10'
    Seq = jnp.ones(n_iter) * 0.001
    target_residual_schedule = jnp.full(n_iter, 10)

    n_samples = 1e4
    heuristic_gaussianMeanField_Nicolas(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq,
                                        target_residual_schedule=target_residual_schedule, title_seq=Seq_title,
                                        OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussianMeanField_Nicolas_u10_fixed1():
    n_iter = int(5e2)
    Seq_title = '1em3_u10'
    Seq = jnp.ones(n_iter) * 0.001
    target_residual_schedule = jnp.full(n_iter, 10)

    n_samples = 1e4
    heuristic_gaussianMeanField_Nicolas(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq,
                                        target_residual_schedule=target_residual_schedule, title_seq=Seq_title,
                                        OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def time_gaussianMeanField_Nicolas_fixed1em3():
    n_iter = int(5e2)
    Seq_title = '1em3'
    Seq = jnp.ones(n_iter) * 0.001
    n_samples = 1e4
    gaussianMeanField_Nicolas(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq, title_seq=Seq_title,
                              OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def time_gaussianMeanField_bj():
    N_iters = 500
    sgd = 1e-3
    num_samples = int(1e4)
    gaussianMeanField_ADVI_blackjax(n_repetitions, N_iters, num_samples, sgd, OUTPUT_PATH=OUTPUT_PATH)

@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def time_ngd_diagonal():
    n_iter = int(5e2)
    n_samples = int(1e4)
    lr = 1 / jnp.arange(1, n_iter + 1)
    ngd_diagonal(keys, n_iter, n_samples, lr, OUTPUT_PATH=OUTPUT_PATH)


if __name__ == "__main__":
    time_heuristic_gaussianMeanField_Nicolas_u10_fixed1em3()
    time_heuristic_gaussianMeanField_Nicolas_u10_fixed1()
    time_gaussianMeanField_Nicolas_fixed1em3()
    time_gaussianMeanField_bj()
    time_ngd_diagonal()